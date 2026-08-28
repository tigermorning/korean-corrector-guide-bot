// 실험: 같은 질문·같은 검색 결과를 로컬 Ollama(qwen3.5:2b)와 Gemini API에
// 각각 흘려보내, 답변 리듬(길이·속도)과 판정 결과가 어떻게 달라지는지 비교한다.
// 노션 "조금 더 발전시키고 싶다면?" 항목("선택 엔진 비교") 대응.
import { readFile, writeFile } from "node:fs/promises";
import { pipeline } from "@huggingface/transformers";
import { buildSearchIndex, hybridSearch, topCosine } from "../lib/search.js";
import * as ollamaEngine from "../lib/ollama.js";
import * as geminiEngine from "../lib/gemini.js";
import { buildSystemPrompt, buildJudgePrompt, normalizeJudgeScore } from "../lib/prompts.js";
import { extractFirstJson } from "../lib/ollama.js";

const QUERY_PREFIX = "task: search result | query: ";
const OLLAMA_MODEL = "qwen3.5:2b";
const GEMINI_MODEL = "gemini-3.6-flash";

const QUESTION_IDS = ["u01", "u06", "u09", "d01", "d08", "d09", "n01", "n02"];

async function loadPicked() {
  const raw = await readFile("data/eval/eval_set.jsonl", "utf-8");
  const all = raw.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => JSON.parse(l));
  return QUESTION_IDS.map((id) => all.find((q) => q.id === id)).filter(Boolean);
}

async function runOne(engineName, { question, sources, weak, chatFn, judgeFn }) {
  const messages = [
    { role: "system", content: buildSystemPrompt(sources, { weak }) },
    { role: "user", content: question },
  ];
  const t0 = Date.now();
  const answer = await chatFn(messages);
  const genMs = Date.now() - t0;

  let verdict = null;
  try {
    const raw = await judgeFn([{ role: "user", content: buildJudgePrompt({ question, answer, sources }) }]);
    const parsed = extractFirstJson(raw) ?? JSON.parse(raw);
    if (parsed) verdict = { ...parsed, score: normalizeJudgeScore(parsed.score) };
  } catch (err) {
    console.error(`  [${engineName}] 판정 실패:`, err.message);
  }

  return { engine: engineName, answer, answerLen: answer.length, genMs, verdict };
}

async function main() {
  const geminiKey = (await readFile("scripts/.gemini_key.local", "utf-8")).trim();
  if (!geminiKey) throw new Error("scripts/.gemini_key.local이 비어 있다");

  const questions = await loadPicked();
  const vectorstore = JSON.parse(await readFile("data/vectorstore.json", "utf-8"));
  const index = buildSearchIndex(vectorstore);
  const extractor = await pipeline("feature-extraction", vectorstore.model, { dtype: vectorstore.dtype || "q4" });

  const rows = [];
  for (const item of questions) {
    const out = await extractor(QUERY_PREFIX + item.question, { pooling: "mean", normalize: true });
    const qv = Array.from(out.data);
    const sources = hybridSearch(index, qv, item.question, { vectorTopN: 10, bm25TopN: 5 });
    const weak = topCosine(sources) < 0.55;

    console.log(`[${item.id}] Ollama 생성 중...`);
    const ollamaResult = await runOne("ollama", {
      question: item.question,
      sources,
      weak,
      chatFn: (messages) => ollamaEngine.chatOnce({ baseUrl: "http://localhost:11434", model: OLLAMA_MODEL, messages }),
      judgeFn: (messages) => ollamaEngine.chatJSON({ baseUrl: "http://localhost:11434", model: OLLAMA_MODEL, messages }),
    });

    console.log(`[${item.id}] Gemini 생성 중...`);
    const geminiResult = await runOne("gemini", {
      question: item.question,
      sources,
      weak,
      chatFn: (messages) => geminiEngine.chatOnce({ apiKey: geminiKey, model: GEMINI_MODEL, messages }),
      judgeFn: (messages) => geminiEngine.chatJSON({ apiKey: geminiKey, model: GEMINI_MODEL, messages }),
    });

    rows.push({ id: item.id, category: item.category, question: item.question, weak, ollama: ollamaResult, gemini: geminiResult });
    console.log(
      `  ollama: ${ollamaResult.genMs}ms, ${ollamaResult.answerLen}자, score=${ollamaResult.verdict?.score}` +
        ` | gemini: ${geminiResult.genMs}ms, ${geminiResult.answerLen}자, score=${geminiResult.verdict?.score}`
    );
  }

  await writeFile("data/eval/engine_comparison.json", JSON.stringify(rows, null, 2), "utf-8");

  const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
  console.log("\n=== 요약 ===");
  console.log(
    `Ollama  평균 응답시간 ${avg(rows.map((r) => r.ollama.genMs)).toFixed(0)}ms, 평균 길이 ${avg(rows.map((r) => r.ollama.answerLen)).toFixed(0)}자, 평균 점수 ${avg(rows.map((r) => r.ollama.verdict?.score ?? 0)).toFixed(1)}`
  );
  console.log(
    `Gemini  평균 응답시간 ${avg(rows.map((r) => r.gemini.genMs)).toFixed(0)}ms, 평균 길이 ${avg(rows.map((r) => r.gemini.answerLen)).toFixed(0)}자, 평균 점수 ${avg(rows.map((r) => r.gemini.verdict?.score ?? 0)).toFixed(1)}`
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
