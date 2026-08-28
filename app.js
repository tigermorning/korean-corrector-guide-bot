import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { loadVectorstore, buildSearchIndex, hybridSearch, topCosine } from "./lib/search.js";
import { streamChat, chatJSON, checkHealth, extractFirstJson } from "./lib/ollama.js";
import { REFUSAL_MARKER, buildSystemPrompt, buildJudgePrompt, normalizeJudgeScore } from "./lib/prompts.js";

// EmbeddingGemma 공식 쿼리 프롬프트(문서 쪽은 scripts/build_vectorstore.mjs의
// documentPrefix와 짝을 이룬다 — 둘 다 안 지키면 벡터 공간이 어긋난다).
const QUERY_PREFIX = "task: search result | query: ";

// 원점수(코사인, 정규화 전) 기준. 이 값보다 낮으면 "약한 근거"로 본다.
// 0.55는 참조 구현(모두콘, 다른 임베딩 모델·코퍼스)의 잠정치이고, 우리는
// EmbeddingGemma+이 68청크 코퍼스로 다시 실측해 잡았다(docs/EXPERIMENTS.md).
// 검색 결과를 버리지 않는다 — 약하면 프롬프트를 보수화하고 UI에 경고만 낸다.
const WEAK_EVIDENCE_THRESHOLD = 0.55;
const VECTOR_TOP_N = 10;
const BM25_TOP_N = 5;

const el = {
  engineStatus: document.getElementById("engineStatus"),
  engineProgressBar: document.getElementById("engineProgressBar"),
  ollamaBanner: document.getElementById("ollamaBanner"),
  ollamaBannerText: document.getElementById("ollamaBannerText"),
  ollamaRetryBtn: document.getElementById("ollamaRetryBtn"),
  chatLog: document.getElementById("chatLog"),
  questionInput: document.getElementById("questionInput"),
  sendBtn: document.getElementById("sendBtn"),
  cancelBtn: document.getElementById("cancelBtn"),
  baseUrlInput: document.getElementById("baseUrlInput"),
  modelInput: document.getElementById("modelInput"),
};

// 설정값은 localStorage에 남겨 새로고침해도 다시 안 적게 한다.
el.baseUrlInput.value = localStorage.getItem("guidebot_base_url") || el.baseUrlInput.value;
el.modelInput.value = localStorage.getItem("guidebot_model") || el.modelInput.value;
el.baseUrlInput.addEventListener("change", () => localStorage.setItem("guidebot_base_url", el.baseUrlInput.value));
el.modelInput.addEventListener("change", () => localStorage.setItem("guidebot_model", el.modelInput.value));

let searchIndex = null;
let extractor = null;
let currentAbort = null;
let engineReady = false;

async function initEngine() {
  try {
    const vectorstore = await loadVectorstore();
    searchIndex = buildSearchIndex(vectorstore);

    extractor = await pipeline("feature-extraction", vectorstore.model, {
      dtype: vectorstore.dtype || "q4",
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          const pct = Math.round((p.loaded / p.total) * 100);
          el.engineProgressBar.style.width = `${pct}%`;
          el.engineStatus.firstChild.textContent = `모델 내려받는 중... ${pct}% (${p.file || ""})`;
        } else if (p.status === "ready" || p.status === "done") {
          el.engineProgressBar.style.width = "100%";
        }
      },
    });

    engineReady = true;
    el.engineStatus.firstChild.textContent = `준비 완료 — 청크 ${vectorstore.chunks.length}개 로드됨 (${vectorstore.dim}차원).`;
    el.questionInput.disabled = false;
    el.sendBtn.disabled = false;
    el.questionInput.focus();

    await refreshOllamaBanner();
  } catch (err) {
    console.error(err);
    el.engineStatus.firstChild.textContent = `엔진 초기화 실패: ${err.message}`;
  }
}

/** GET /api/tags로 Ollama 상태를 확인해, "엔진 꺼짐"과 "생성 실패"를 구분한다. */
async function refreshOllamaBanner() {
  const baseUrl = el.baseUrlInput.value.trim();
  const health = await checkHealth(baseUrl);
  if (health.ok) {
    el.ollamaBanner.hidden = true;
  } else {
    el.ollamaBanner.hidden = false;
    el.ollamaBannerText.textContent = `로컬 Ollama(${baseUrl})에 연결할 수 없습니다 (${health.reason}). ollama serve가 실행 중인지, CORS(OLLAMA_ORIGINS) 설정이 됐는지 확인하세요.`;
  }
  return health.ok;
}

el.ollamaRetryBtn.addEventListener("click", refreshOllamaBanner);

function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  el.chatLog.appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
  return div;
}

function renderSources(container, sources) {
  const wrap = document.createElement("div");
  wrap.className = "sources";
  const snippets = document.createElement("div");
  sources.forEach((s) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    const methodLabel = s.method === "vector" ? "의미" : "어휘(BM25)";
    chip.textContent = `${s.section.split(">").pop().trim()} · ${methodLabel} ${s.score.toFixed(2)}`;
    const snippet = document.createElement("div");
    snippet.className = "snippet";
    snippet.innerHTML = `<b>[${s.id}] ${s.section}</b> — ${methodLabel} 검색, 점수 ${s.score.toFixed(3)}<br>${s.text}<br><a href="${s.url}" target="_blank" rel="noopener">원문 보기 ↗</a>`;
    chip.addEventListener("click", () => snippet.classList.toggle("open"));
    wrap.appendChild(chip);
    snippets.appendChild(snippet);
  });
  container.appendChild(wrap);
  container.appendChild(snippets);
}

/** 근거성과 자연스러움은 별개다 — fluency는 항상 뜨는 고정 안내, 나머지는 실제 판정. */
function renderBadges(container, { weak, verdict, pending, judgeFailed }) {
  const wrap = document.createElement("div");
  wrap.className = "badges";

  const fluency = document.createElement("span");
  fluency.className = "badge fluency";
  fluency.textContent = "AI가 생성한 자연스러운 문장 (근거성과 별개)";
  wrap.appendChild(fluency);

  if (weak) {
    const w = document.createElement("span");
    w.className = "badge grounded-medium";
    w.textContent = "⚠ 약한 근거 — 검색 유사도가 낮습니다";
    wrap.appendChild(w);
  }

  if (pending) {
    const p = document.createElement("span");
    p.className = "badge pending";
    p.textContent = "판정 중...";
    wrap.appendChild(p);
  } else if (judgeFailed) {
    const p = document.createElement("span");
    p.className = "badge pending";
    p.textContent = "판정 실패 (답변은 유지됩니다)";
    wrap.appendChild(p);
  } else if (verdict) {
    const fields = [
      ["grounded", "근거 일치"],
      ["noHalluc", "환각 없음"],
      ["cited", "인용 표시"],
      ["refusal", "정당한 거절"],
    ];
    for (const [key, label] of fields) {
      const b = document.createElement("span");
      const val = verdict[key];
      b.className = `badge ${val ? "field-true" : "field-false"}`;
      b.textContent = `${label} ${val ? "✓" : "✗"}`;
      wrap.appendChild(b);
    }
    const score = document.createElement("span");
    score.className = "badge field-score";
    score.textContent = `점수 ${verdict.score}/100`;
    wrap.appendChild(score);
    if (verdict.comment) {
      const comment = document.createElement("span");
      comment.className = "badge";
      comment.textContent = verdict.comment;
      wrap.appendChild(comment);
    }
  }

  container.appendChild(wrap);
  return wrap;
}

function renderFeedback(container, record) {
  const wrap = document.createElement("div");
  wrap.className = "feedback";
  const label = document.createElement("span");
  label.textContent = "이 답변 도움됐나요?";
  const up = document.createElement("button");
  up.textContent = "👍";
  const down = document.createElement("button");
  down.textContent = "👎";
  const saved = document.createElement("span");

  function save(vote) {
    up.classList.toggle("active", vote === "up");
    down.classList.toggle("active", vote === "down");
    const log = JSON.parse(localStorage.getItem("guidebot_feedback_log") || "[]");
    log.push({ ...record, vote, ts: new Date().toISOString() });
    localStorage.setItem("guidebot_feedback_log", JSON.stringify(log));
    saved.textContent = "저장됨 (이 브라우저에만)";
  }

  up.addEventListener("click", () => save("up"));
  down.addEventListener("click", () => save("down"));
  wrap.append(label, up, down, saved);
  container.appendChild(wrap);
}

async function judgeAnswer({ baseUrl, model, question, answer, sources }) {
  const judgePrompt = buildJudgePrompt({ question, answer, sources });
  const raw = await chatJSON({ baseUrl, model, messages: [{ role: "user", content: judgePrompt }] });
  const parsed = extractFirstJson(raw) ?? JSON.parse(raw);
  if (!parsed) return null;
  return { ...parsed, score: normalizeJudgeScore(parsed.score) };
}

async function handleAsk(question) {
  const baseUrl = el.baseUrlInput.value.trim();
  const model = el.modelInput.value.trim();
  if (!model) {
    appendMessage("bot", "먼저 위 '엔진 설정'에서 Ollama 모델 이름을 입력하세요.");
    return;
  }
  if (!(await refreshOllamaBanner())) {
    return; // 배너가 이미 상태를 보여준다 — 생성 실패로 오해하지 않도록 여기서 멈춘다.
  }

  appendMessage("user", question);
  const botDiv = appendMessage("bot", "");

  const queryOutput = await extractor(QUERY_PREFIX + question, { pooling: "mean", normalize: true });
  const queryVector = Array.from(queryOutput.data);
  const sources = hybridSearch(searchIndex, queryVector, question, {
    vectorTopN: VECTOR_TOP_N,
    bm25TopN: BM25_TOP_N,
  });
  const weak = topCosine(sources) < WEAK_EVIDENCE_THRESHOLD;

  el.sendBtn.disabled = true;
  el.questionInput.disabled = true;
  el.cancelBtn.style.display = "inline-block";
  currentAbort = new AbortController();

  let answer = "";
  try {
    const messages = [
      { role: "system", content: buildSystemPrompt(sources, { weak }) },
      { role: "user", content: question },
    ];
    for await (const delta of streamChat({ baseUrl, model, messages, signal: currentAbort.signal })) {
      answer += delta;
      botDiv.textContent = answer;
    }
  } catch (err) {
    if (err.name === "AbortError") {
      answer += "\n\n[사용자가 취소함]";
      botDiv.textContent = answer; // 이미 받은 답변은 지우지 않는다.
    } else {
      // 스트리밍 오류가 나도 이미 받은 답변은 유지한다.
      botDiv.textContent = (answer ? answer + "\n\n" : "") + `[답변 생성 중 오류: ${err.message}]`;
      el.sendBtn.disabled = false;
      el.questionInput.disabled = false;
      el.cancelBtn.style.display = "none";
      renderSources(botDiv, sources);
      return;
    }
  }

  if (weak) botDiv.classList.add("weak-evidence");
  renderSources(botDiv, sources);
  renderBadges(botDiv, { weak, pending: true });
  const isRefusal = answer.includes(REFUSAL_MARKER);
  renderFeedback(botDiv, { question, answer, weak, isRefusal, sources: sources.map((s) => s.id) });

  el.sendBtn.disabled = false;
  el.questionInput.disabled = false;
  el.cancelBtn.style.display = "none";
  currentAbort = null;

  const badgesRow = botDiv.querySelector(".badges");
  try {
    const verdict = await judgeAnswer({ baseUrl, model, question, answer, sources });
    // refusal 필드는 판정 모델의 자기 보고보다 실제 답변 텍스트를 더 신뢰한다
    // (작은 모델이 이 판단을 틀리는 경우가 실측으로 확인됨, docs/EXPERIMENTS.md).
    const corrected = verdict ? { ...verdict, refusal: isRefusal } : null;
    badgesRow.replaceWith(renderBadges(botDiv, { weak, verdict: corrected, judgeFailed: !verdict }));
  } catch (err) {
    console.error("judge failed", err);
    badgesRow.replaceWith(renderBadges(botDiv, { weak, judgeFailed: true }));
  }
}

el.sendBtn.addEventListener("click", () => {
  if (!engineReady) return;
  const q = el.questionInput.value.trim();
  if (!q) return;
  el.questionInput.value = "";
  handleAsk(q);
});

el.questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") el.sendBtn.click();
});

el.cancelBtn.addEventListener("click", () => {
  currentAbort?.abort();
});

initEngine();
