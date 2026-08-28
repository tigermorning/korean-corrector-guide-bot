// 청크 JSON(data/chunks/*.json)을 읽어 각 청크를 임베딩하고
// data/vectorstore.json(정적 파일)으로 합친다. 브라우저 런타임(같은 모델,
// transformers.js)이 질문을 임베딩할 때와 같은 벡터 공간을 쓰도록, 반드시
// 같은 모델·같은 dtype·같은 프롬프트 규칙(EmbeddingGemma 공식 템플릿)을
// 지켜야 한다 — app.js의 query 접두사와 여기 document 접두사가 한 쌍이다.

import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { pipeline } from "@huggingface/transformers";

const MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX";
const DTYPE = "q4";
// EmbeddingGemma 공식 프롬프트 템플릿(문서용). section을 title 자리에 넣으면
// 성능이 오른다고 모델 카드가 명시한다 — query 쪽은 app.js에서 같은 규칙으로
// "task: search result | query: "를 붙인다.
const documentPrefix = (title) => `title: ${title || "none"} | text: `;
const CHUNKS_DIR = path.resolve("data/chunks");
const OUT_FILE = path.resolve("data/vectorstore.json");

async function loadAllChunks() {
  const files = (await readdir(CHUNKS_DIR)).filter((f) => f.endsWith(".json"));
  const chunks = [];
  for (const file of files) {
    const raw = await readFile(path.join(CHUNKS_DIR, file), "utf-8");
    const items = JSON.parse(raw);
    for (const item of items) {
      chunks.push({ ...item, source_file: file });
    }
  }
  return chunks;
}

function meanPool(output) {
  // transformers.js feature-extraction 파이프라인에 pooling:"mean", normalize:true를
  // 넘기면 이미 풀링·정규화된 벡터가 나오므로 이 함수는 방어적 폴백일 뿐이다.
  return Array.from(output.data);
}

async function main() {
  const chunks = await loadAllChunks();
  const ids = new Set();
  for (const c of chunks) {
    if (ids.has(c.id)) throw new Error(`중복 id: ${c.id}`);
    ids.add(c.id);
  }
  console.log(`청크 ${chunks.length}개 로드. 모델 ${MODEL_ID}(dtype=${DTYPE}) 준비 중...`);

  const extractor = await pipeline("feature-extraction", MODEL_ID, { dtype: DTYPE });

  const vectorstore = [];
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    const input = documentPrefix(c.section) + c.text;
    const output = await extractor(input, { pooling: "mean", normalize: true });
    const vector = meanPool(output);
    vectorstore.push({
      id: c.id,
      url: c.url,
      section: c.section,
      text: c.text,
      source_file: c.source_file,
      vector,
    });
    if ((i + 1) % 10 === 0 || i === chunks.length - 1) {
      console.log(`  임베딩 ${i + 1}/${chunks.length}`);
    }
  }

  await writeFile(
    OUT_FILE,
    JSON.stringify({ model: MODEL_ID, dtype: DTYPE, dim: vectorstore[0].vector.length, chunks: vectorstore }, null, 0),
    "utf-8"
  );
  console.log(`완료: ${OUT_FILE} (${vectorstore.length}개 벡터, 차원 ${vectorstore[0].vector.length})`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
