// 실험: 쿼리 확장(query expansion)이 실제로 검색을 개선하는지 — 이상하고
// 구어체인 질문 세트(data/eval/hard_questions.jsonl, 기존 25문항과 겹치지 않음)로
// 측정한다. 바꾸는 변수는 "쿼리 확장 유무" 하나뿐이고, 검색 설정·모델은 고정.
import { readFile, writeFile } from "node:fs/promises";
import { pipeline } from "@huggingface/transformers";
import { buildSearchIndex, hybridSearch, topCosine } from "../lib/search.js";
import { chatOnce } from "../lib/ollama.js";

const BASE_URL = "http://localhost:11434";
const MODEL = "qwen3.5:2b";
const QUERY_PREFIX = "task: search result | query: ";

async function loadHardQuestions() {
  const raw = await readFile("data/eval/hard_questions.jsonl", "utf-8");
  return raw.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => JSON.parse(l));
}

async function expandQuery(question) {
  const prompt = [
    "다음 질문을 다른 표현으로 정확히 2개만 다시 써줘.",
    "설명이나 번호, 따옴표 없이 질문만 한 줄씩 출력해.",
    `질문: ${question}`,
  ].join("\n");
  const raw = await chatOnce({ baseUrl: BASE_URL, model: MODEL, messages: [{ role: "user", content: prompt }] });
  return raw
    .split("\n")
    .map((l) => l.replace(/^[-*\d.\s]+/, "").trim())
    .filter(Boolean)
    .slice(0, 2);
}

function mergeByMaxCosine(variantResultsList, chunks) {
  // 각 변형(원 질문 포함)의 코사인 중 청크별 최댓값을 취한다.
  const maxCosine = new Map();
  for (const cosineArr of variantResultsList) {
    chunks.forEach((c, i) => {
      const prev = maxCosine.get(c.id) ?? -1;
      if (cosineArr[i] > prev) maxCosine.set(c.id, cosineArr[i]);
    });
  }
  return maxCosine;
}

async function main() {
  const questions = await loadHardQuestions();
  const store = JSON.parse(await readFile("data/vectorstore.json", "utf-8"));
  const index = buildSearchIndex(store);
  const extractor = await pipeline("feature-extraction", store.model, { dtype: store.dtype || "q4" });
  const chunks = store.chunks;

  async function embed(text) {
    const out = await extractor(QUERY_PREFIX + text, { pooling: "mean", normalize: true });
    return Array.from(out.data);
  }
  function cosineAll(qv) {
    return chunks.map((c) => {
      let dot = 0;
      for (let i = 0; i < qv.length; i++) dot += qv[i] * c.vector[i];
      return dot;
    });
  }

  const rows = [];
  for (const item of questions) {
    // 기준: 원 질문 그대로 하이브리드 검색
    const baseline = hybridSearch(index, await embed(item.question), item.question, { vectorTopN: 10, bm25TopN: 5 });
    const baselineIds = baseline.map((r) => r.id);
    const baselineHit = item.gold_chunk_ids.length ? item.gold_chunk_ids.some((g) => baselineIds.includes(g)) : null;

    // 실험: LLM으로 패러프레이즈 2개 생성 -> 3개 벡터의 청크별 최댓값 코사인으로 재검색
    const paraphrases = await expandQuery(item.question);
    const variantCosines = [cosineAll(await embed(item.question))];
    for (const p of paraphrases) variantCosines.push(cosineAll(await embed(p)));
    const maxCosine = mergeByMaxCosine(variantCosines, chunks);
    const expandedTop10 = [...maxCosine.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10).map(([id]) => id);
    const expandedHit = item.gold_chunk_ids.length ? item.gold_chunk_ids.some((g) => expandedTop10.includes(g)) : null;

    rows.push({
      id: item.id,
      question: item.question,
      paraphrases,
      gold: item.gold_chunk_ids,
      baseline_top1_cosine: topCosine(baseline),
      baseline_hit: baselineHit,
      expanded_top1_cosine: Math.max(...maxCosine.values()),
      expanded_hit: expandedHit,
    });
    console.log(`[${item.id}] baseline_hit=${baselineHit} expanded_hit=${expandedHit} (${item.question})`);
  }

  await writeFile("data/eval/query_expansion_results.json", JSON.stringify(rows, null, 2), "utf-8");

  const withGold = rows.filter((r) => r.gold.length);
  const baselineHitRate = withGold.filter((r) => r.baseline_hit).length / withGold.length;
  const expandedHitRate = withGold.filter((r) => r.expanded_hit).length / withGold.length;
  console.log(`\n기준 hitRate=${baselineHitRate.toFixed(2)}  확장 hitRate=${expandedHitRate.toFixed(2)} (gold 있는 문항 ${withGold.length}개 기준)`);
}

main().catch((e) => { console.error(e); process.exit(1); });
