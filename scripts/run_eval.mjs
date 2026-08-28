// 실험·평가 스크립트.
// Part A: 검색 설정(vectorTopN·bm25TopN)별 retrieval 성능 비교 — Ollama 없이 빠르게.
// Part B: 기본 설정으로 실제 생성+판정까지 돌려 결과를 JSON으로 남긴다(사람이
// docs/RUBRIC.md 기준으로 채점할 원본 자료).

import { readFile, writeFile } from "node:fs/promises";
import { pipeline } from "@huggingface/transformers";
import { buildSearchIndex, hybridSearch, topCosine } from "../lib/search.js";
import { streamChat, chatJSON, extractFirstJson } from "../lib/ollama.js";
import { buildSystemPrompt, buildJudgePrompt, normalizeJudgeScore, REFUSAL_MARKER } from "../lib/prompts.js";

const BASE_URL = "http://localhost:11434";
const MODEL = "qwen3.5:2b";
const QUERY_PREFIX = "task: search result | query: ";
const WEAK_EVIDENCE_THRESHOLD = 0.55;

async function loadEvalSet() {
  const raw = await readFile("data/eval/eval_set.jsonl", "utf-8");
  return raw
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l));
}

async function embedAll(extractor, questions) {
  const vectors = new Map();
  for (const q of questions) {
    const out = await extractor(QUERY_PREFIX + q.question, { pooling: "mean", normalize: true });
    vectors.set(q.id, Array.from(out.data));
  }
  return vectors;
}

function retrievalMetrics(evalSet, queryVectors, index, opts) {
  const rows = [];
  for (const item of evalSet) {
    const qv = queryVectors.get(item.id);
    const results = hybridSearch(index, qv, item.question, opts);
    const retrievedIds = results.map((r) => r.id);
    const top = topCosine(results);
    let hit = null;
    let rank = null;
    if (item.gold_chunk_ids.length > 0) {
      hit = item.gold_chunk_ids.some((g) => retrievedIds.includes(g));
      for (let i = 0; i < retrievedIds.length; i++) {
        if (item.gold_chunk_ids.includes(retrievedIds[i])) {
          rank = i + 1;
          break;
        }
      }
    }
    const wouldFlagWeak = top < WEAK_EVIDENCE_THRESHOLD;
    rows.push({ id: item.id, category: item.category, hit, rank, topCosine: top, wouldFlagWeak, expect_refusal: item.expect_refusal });
  }
  return rows;
}

function summarize(rows) {
  const withGold = rows.filter((r) => r.hit !== null);
  const hitRate = withGold.length ? withGold.filter((r) => r.hit).length / withGold.length : null;
  const ranks = withGold.filter((r) => r.rank !== null).map((r) => r.rank);
  const avgRank = ranks.length ? ranks.reduce((a, b) => a + b, 0) / ranks.length : null;
  const oos = rows.filter((r) => r.expect_refusal);
  const weakFlagRate = oos.length ? oos.filter((r) => r.wouldFlagWeak).length / oos.length : null;
  return { n: rows.length, hitRate, avgRank, weakFlagRate, oosCount: oos.length };
}

async function partA(evalSet, queryVectors, index) {
  const settings = [
    { name: "vector만 (top10)", vectorTopN: 10, bm25TopN: 0 },
    { name: "하이브리드 (vector10+bm25 5) [기본값]", vectorTopN: 10, bm25TopN: 5 },
    { name: "하이브리드 (vector5+bm25 3)", vectorTopN: 5, bm25TopN: 3 },
    { name: "하이브리드 (vector10+bm25 10)", vectorTopN: 10, bm25TopN: 10 },
  ];
  const results = [];
  for (const s of settings) {
    const rows = retrievalMetrics(evalSet, queryVectors, index, { vectorTopN: s.vectorTopN, bm25TopN: s.bm25TopN });
    results.push({ setting: s.name, ...summarize(rows), rows });
  }
  return results;
}

async function partB(evalSet, queryVectors, index) {
  const rows = [];
  for (const item of evalSet) {
    const qv = queryVectors.get(item.id);
    const sources = hybridSearch(index, qv, item.question, { vectorTopN: 10, bm25TopN: 5 });
    const weak = topCosine(sources) < WEAK_EVIDENCE_THRESHOLD;

    const messages = [
      { role: "system", content: buildSystemPrompt(sources, { weak }) },
      { role: "user", content: item.question },
    ];
    const answer = await (async () => {
      let full = "";
      for await (const delta of streamChat({ baseUrl: BASE_URL, model: MODEL, messages })) full += delta;
      return full;
    })();

    const isRefusal = answer.includes(REFUSAL_MARKER);
    let verdict = null;
    try {
      const raw = await chatJSON({
        baseUrl: BASE_URL,
        model: MODEL,
        messages: [{ role: "user", content: buildJudgePrompt({ question: item.question, answer, sources }) }],
      });
      const parsed = extractFirstJson(raw) ?? JSON.parse(raw);
      if (parsed) verdict = { ...parsed, score: normalizeJudgeScore(parsed.score), refusal: isRefusal };
    } catch (err) {
      console.error(`  [${item.id}] 판정 실패:`, err.message);
    }

    rows.push({
      id: item.id,
      category: item.category,
      question: item.question,
      expect_refusal: item.expect_refusal,
      weak_evidence_flag: weak,
      answer,
      is_refusal: isRefusal,
      sources: sources.map((s) => ({ id: s.id, method: s.method, score: s.score })),
      judge: verdict,
    });
    console.log(`  [${item.id}] 완료 (weak=${weak})`);
  }
  return rows;
}

async function main() {
  const evalSet = await loadEvalSet();
  const vectorstore = JSON.parse(await readFile("data/vectorstore.json", "utf-8"));
  const index = buildSearchIndex(vectorstore);

  console.log(`평가셋 ${evalSet.length}건 로드. 모델 ${vectorstore.model} 준비 중...`);
  const extractor = await pipeline("feature-extraction", vectorstore.model, { dtype: vectorstore.dtype || "q4" });
  const queryVectors = await embedAll(extractor, evalSet);

  console.log("Part A: 검색 설정 비교 중...");
  const partAResults = await partA(evalSet, queryVectors, index);

  console.log("Part B: 생성+판정 전체 파이프라인 실행 중 (Ollama 호출, 시간 걸림)...");
  const partBResults = await partB(evalSet, queryVectors, index);

  const out = {
    generatedAt: "2026-08-28",
    embeddingModel: vectorstore.model,
    llmModel: MODEL,
    weakEvidenceThreshold: WEAK_EVIDENCE_THRESHOLD,
    partA: partAResults,
    partB: partBResults,
  };
  await writeFile("data/eval/results.json", JSON.stringify(out, null, 2), "utf-8");
  console.log("완료: data/eval/results.json");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
