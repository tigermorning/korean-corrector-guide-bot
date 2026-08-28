// 하이브리드 검색: 코사인 유사도(임베딩) 상위 N개 + 아직 안 들어온 청크 중
// BM25(어휘 매칭) 상위 M개를 합친다(중복 제거). 가중합이 아니라 "서로 다른
// 실패 방식을 가진 두 관찰 장치"로 배치하는 방식 — 벡터가 못 잡는 고유명사·
// 정확한 표기를 BM25가 보충한다.
//
// 벡터스토어는 Node 스크립트(scripts/build_vectorstore.mjs)가 미리 만든 정적
// 파일이고, 질문 임베딩만 브라우저에서 그 자리에서 계산한다(같은 모델·같은
// dtype·같은 EmbeddingGemma 프롬프트 규칙을 반드시 지켜야 벡터 공간이
// 어긋나지 않는다 — app.js의 query 접두사 참고).

import { buildBM25Index, bm25Scores } from "./bm25.js";

export async function loadVectorstore(url = "data/vectorstore.json") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`벡터스토어 로드 실패: ${res.status}`);
  return res.json();
}

export function buildSearchIndex(vectorstore) {
  const bm25 = buildBM25Index(vectorstore.chunks.map((c) => ({ id: c.id, text: c.text })));
  return { vectorstore, bm25 };
}

function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // 벡터가 이미 정규화되어 있어 내적이 곧 코사인 유사도
}

/**
 * @param {object} index buildSearchIndex()의 결과
 * @param {number[]} queryVector 질문 벡터(EmbeddingGemma query 접두사 적용된 것)
 * @param {string} queryText 원문 질문(BM25용)
 * @param {{vectorTopN?: number, bm25TopN?: number}} opts
 * @returns {{id,url,section,text,cosine,bm25Raw,method,score}[]}
 *   method는 "vector" 또는 "bm25". score는 방법별 의미가 다르다 —
 *   vector 항목은 원점수 코사인, bm25 항목은 이번 검색 내 최대 BM25로
 *   정규화한 0~1 값이다. 최고 근거 신뢰도(약한 근거 판정)는 항상 cosine
 *   필드(원점수)로만 본다 — 정규화된 score를 섞어 쓰면 안 된다.
 */
export function hybridSearch(index, queryVector, queryText, opts = {}) {
  const vectorTopN = opts.vectorTopN ?? 10;
  const bm25TopN = opts.bm25TopN ?? 5;
  const { chunks } = index.vectorstore;

  const cosineRaw = chunks.map((c) => cosine(queryVector, c.vector));
  const bm25Raw = bm25Scores(index.bm25, queryText);
  const bm25Max = Math.max(...bm25Raw, 1e-9);

  const vectorRanked = chunks
    .map((c, i) => ({ chunk: c, cosine: cosineRaw[i], bm25Raw: bm25Raw[i] }))
    .sort((a, b) => b.cosine - a.cosine)
    .slice(0, vectorTopN)
    .map((r) => ({ ...r, method: "vector", score: r.cosine }));

  const vectorIds = new Set(vectorRanked.map((r) => r.chunk.id));

  const bm25Ranked = chunks
    .map((c, i) => ({ chunk: c, cosine: cosineRaw[i], bm25Raw: bm25Raw[i] }))
    .filter((r) => !vectorIds.has(r.chunk.id) && r.bm25Raw > 0)
    .sort((a, b) => b.bm25Raw - a.bm25Raw)
    .slice(0, bm25TopN)
    .map((r) => ({ ...r, method: "bm25", score: r.bm25Raw / bm25Max }));

  const merged = [...vectorRanked, ...bm25Ranked].map((r) => ({
    id: r.chunk.id,
    url: r.chunk.url,
    section: r.chunk.section,
    text: r.chunk.text,
    cosine: r.cosine,
    bm25Raw: r.bm25Raw,
    method: r.method,
    score: r.score,
  }));

  return merged;
}

/** 이번 검색이 근거로 삼기에 충분한지 — 항상 원점수 코사인 최댓값으로 본다. */
export function topCosine(results) {
  return results.reduce((max, r) => Math.max(max, r.cosine), 0);
}
