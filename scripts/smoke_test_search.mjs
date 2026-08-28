// 임시 확인용: 벡터스토어가 실제로 의미 검색이 되는지 질문 몇 개로 눈으로 확인.
import { readFile } from "node:fs/promises";
import { pipeline } from "@huggingface/transformers";
import { buildSearchIndex, hybridSearch, topCosine } from "../lib/search.js";

const QUERY_PREFIX = "task: search result | query: ";

async function main() {
  const store = JSON.parse(await readFile("data/vectorstore.json", "utf-8"));
  const index = buildSearchIndex(store);
  const extractor = await pipeline("feature-extraction", store.model, { dtype: store.dtype || "q4" });

  const questions = [
    "자막이랑 일반 글은 왜 다르게 처리해?",
    "왜 사투리는 자동으로 안 고쳐줘?",
    "이 표기는 왜 플래그만 뜨고 안 고쳐져?",
    "구두점 표기는 뭘 골라야 돼?",
    "오늘 서울 날씨 어때?",
  ];

  for (const q of questions) {
    const out = await extractor(QUERY_PREFIX + q, { pooling: "mean", normalize: true });
    const qv = Array.from(out.data);
    const results = hybridSearch(index, qv, q, { vectorTopN: 10, bm25TopN: 5 });
    console.log(`\nQ: ${q}  (top cosine=${topCosine(results).toFixed(3)})`);
    for (const s of results.slice(0, 5)) {
      console.log(`  ${s.method.padEnd(6)} ${s.score.toFixed(3)}  ${s.id}  (${s.section})`);
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
