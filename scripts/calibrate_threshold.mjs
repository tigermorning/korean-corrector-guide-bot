import { readFile } from "node:fs/promises";
import { pipeline } from "@huggingface/transformers";
import { buildSearchIndex, hybridSearch, topCosine } from "../lib/search.js";

const QUERY_PREFIX = "task: search result | query: ";

const store = JSON.parse(await readFile("data/vectorstore.json", "utf-8"));
const index = buildSearchIndex(store);
const extractor = await pipeline("feature-extraction", store.model, { dtype: store.dtype || "q4" });

const qs = [
  ["오늘 서울 날씨 어때?", "무관"],
  ["김치찌개 레시피 알려줘", "무관"],
  ["비트코인 시세 어때?", "무관"],
  ["왜 로컬에서만 실행해야 돼?", "관련(예전 오차단)"],
  ["언어 모델 패스가 뭐야?", "관련(예전 오차단)"],
  ["왜 AI로 고치지 않고 규칙 기반으로만 만들었어?", "관련(예전 검색실패)"],
  ["고유명사 띄어쓰기가 문서 안에서 섞여 있으면 어떻게 처리해?", "관련(예전 검색실패)"],
];
for (const [q, tag] of qs) {
  const out = await extractor(QUERY_PREFIX + q, { pooling: "mean", normalize: true });
  const qv = Array.from(out.data);
  const results = hybridSearch(index, qv, q, { vectorTopN: 10, bm25TopN: 5 });
  console.log(`[${tag}] ${q} -> top cosine=${topCosine(results).toFixed(3)}  top1=${results[0].id}`);
}
