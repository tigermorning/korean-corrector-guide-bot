// 가벼운 BM25 구현. 형태소 분석기 없이 브라우저에서만 돌아야 하므로
// 공백 토큰 + 한글 글자 바이그램을 섞어 쓴다 — 조사가 붙어도("사투리는" vs
// "사투리가") 바이그램이 겹쳐서 완전히 못 찾는 사고를 줄인다.

function tokenize(text) {
  const cleaned = text.replace(/\s+/g, "");
  const wsTokens = text.split(/\s+/).filter(Boolean);
  const bigrams = [];
  for (let i = 0; i < cleaned.length - 1; i++) {
    bigrams.push(cleaned.slice(i, i + 2));
  }
  return [...wsTokens, ...bigrams];
}

const K1 = 1.5;
const B = 0.75;

export function buildBM25Index(docs) {
  // docs: [{id, text}]
  const tokenized = docs.map((d) => tokenize(d.text));
  const df = new Map();
  let totalLen = 0;
  tokenized.forEach((tokens) => {
    totalLen += tokens.length;
    new Set(tokens).forEach((t) => df.set(t, (df.get(t) || 0) + 1));
  });
  const N = docs.length;
  const avgdl = totalLen / N;
  const idf = new Map();
  for (const [term, freq] of df.entries()) {
    idf.set(term, Math.log((N - freq + 0.5) / (freq + 0.5) + 1));
  }
  return { docs, tokenized, idf, avgdl, N };
}

export function bm25Scores(index, query) {
  const qTokens = tokenize(query);
  const scores = new Array(index.N).fill(0);
  for (let i = 0; i < index.N; i++) {
    const tokens = index.tokenized[i];
    const dl = tokens.length;
    const tf = new Map();
    tokens.forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
    let score = 0;
    for (const qt of qTokens) {
      const f = tf.get(qt);
      if (!f) continue;
      const idf = index.idf.get(qt) || 0;
      score += (idf * (f * (K1 + 1))) / (f + K1 * (1 - B + (B * dl) / index.avgdl));
    }
    scores[i] = score;
  }
  return scores;
}
