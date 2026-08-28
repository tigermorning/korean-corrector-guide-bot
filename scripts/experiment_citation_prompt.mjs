// 프롬프트 실험: 인용 지시 문구만 바꾼다(그 외 검색 설정·모델·질문 세트는 고정).
// 기준(A): "근거가 된 조각의 [id]를 답 안에서 표시하세요(예: ...)." — 1회성 부탁.
// 실험(B): 문장마다 반드시 표시하라는 강한 지시 + 예시 문장 포함.
import { readFile, writeFile } from "node:fs/promises";
import { pipeline } from "@huggingface/transformers";
import { buildSearchIndex, hybridSearch, topCosine } from "../lib/search.js";
import { streamChat, chatJSON, extractFirstJson } from "../lib/ollama.js";
import { buildJudgePrompt, normalizeJudgeScore, REFUSAL_MARKER, nowKST } from "../lib/prompts.js";

const BASE_URL = "http://localhost:11434";
const MODEL = "qwen3.5:2b";
const QUERY_PREFIX = "task: search result | query: ";

function buildSystemPromptB(sources, opts = {}) {
  const context = sources.map((s) => `[${s.id} | ${s.section}]\n${s.text}`).join("\n\n");
  const lines = [
    "다음 자료는 한국어 교정기(korean-subtitle-corrector)의 사용법과 설계 배경을 다루는",
    "공개 문서에서 뽑은 조각입니다. 이 자료 안의 내용만 근거로 답하고,",
    "자료에 없는 내용은 절대 만들어내지 마세요.",
  ];
  if (opts.weak) {
    lines.push(
      "",
      "주의: 검색된 조각의 유사도가 낮습니다. 질문과 완전히 맞는 근거가 아닐 수 있으니,",
      "근거에 있는 내용만 짧게 답하고 자료에 없는 부분은 없다고 말합니다."
    );
  } else {
    lines.push("", "근거로 답할 수 없으면 '문서에서 확인되지 않습니다'라고 솔직히 답하세요.");
  }
  lines.push(
    "",
    "**중요 — 인용 규칙**: 답변의 각 문장 또는 각 항목 끝에 그 내용의 근거가 된",
    "조각의 [id]를 반드시 표시하세요. 예: '자막 모드에서는 편집 관례가 적용됩니다 [usage-index-doctype-default].'",
    "[id] 표시가 없는 문장은 근거 없는 주장으로 간주되니 절대 빠뜨리지 마세요.",
    `현재 시각은 ${nowKST()}입니다. '지금', '오늘', '이번 주' 같은 상대 표현은 이 시각을 기준으로 해석합니다.`,
    "한국어로, 간결하게 답하세요.",
    "",
    "[자료]",
    context
  );
  return lines.join("\n");
}

async function main() {
  const evalSet = (await readFile("data/eval/eval_set.jsonl", "utf-8"))
    .split("\n").map((l) => l.trim()).filter(Boolean).map((l) => JSON.parse(l))
    .filter((q) => q.category !== "out_of_scope"); // 인용 실험은 답할 수 있는 질문만

  const vectorstore = JSON.parse(await readFile("data/vectorstore.json", "utf-8"));
  const index = buildSearchIndex(vectorstore);
  const extractor = await pipeline("feature-extraction", vectorstore.model, { dtype: vectorstore.dtype || "q4" });

  const rows = [];
  for (const item of evalSet) {
    const out = await extractor(QUERY_PREFIX + item.question, { pooling: "mean", normalize: true });
    const qv = Array.from(out.data);
    const sources = hybridSearch(index, qv, item.question, { vectorTopN: 10, bm25TopN: 5 });
    const weak = topCosine(sources) < 0.55;

    const messages = [
      { role: "system", content: buildSystemPromptB(sources, { weak }) },
      { role: "user", content: item.question },
    ];
    let answer = "";
    for await (const delta of streamChat({ baseUrl: BASE_URL, model: MODEL, messages })) answer += delta;

    const isRefusal = answer.includes(REFUSAL_MARKER);
    let verdict = null;
    try {
      const raw = await chatJSON({
        baseUrl: BASE_URL, model: MODEL,
        messages: [{ role: "user", content: buildJudgePrompt({ question: item.question, answer, sources }) }],
      });
      const parsed = extractFirstJson(raw) ?? JSON.parse(raw);
      if (parsed) verdict = { ...parsed, score: normalizeJudgeScore(parsed.score), refusal: isRefusal };
    } catch (e) { console.error(item.id, "judge failed", e.message); }

    rows.push({ id: item.id, question: item.question, answer, judge: verdict });
    console.log(`[${item.id}] cited=${verdict?.cited} grounded=${verdict?.grounded} score=${verdict?.score}`);
  }

  await writeFile("data/eval/citation_experiment_variantB.json", JSON.stringify(rows, null, 2), "utf-8");

  const cited = rows.filter((r) => r.judge?.cited).length;
  const grounded = rows.filter((r) => r.judge?.grounded).length;
  const avgScore = rows.reduce((a, r) => a + (r.judge?.score ?? 0), 0) / rows.length;
  console.log(`\n변형 B — cited ${cited}/${rows.length}, grounded ${grounded}/${rows.length}, 평균점수 ${avgScore.toFixed(1)}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
