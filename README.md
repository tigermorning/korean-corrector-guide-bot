# 한국어 교정기 사용 안내 챗봇

`korean-subtitle-corrector`(한국어 띄어쓰기·맞춤법 자동 교정 도구)의 사용법과 설계
배경을 공개 문서 근거로 답하는 RAG 안내 챗봇. 서버가 답을 만들지 않는다 — 브라우저가
임베딩·검색을 전부 수행하고, 페이지를 연 사람 컴퓨터의 로컬 Ollama(`qwen3.5:2b`)가
답을 생성·판정한다.

자세한 기획은 [PRD.md](./PRD.md), 실험 결과는 [docs/EXPERIMENTS.md](./docs/EXPERIMENTS.md),
채점 기준은 [docs/RUBRIC.md](./docs/RUBRIC.md) 참고.

## 배포 주소

https://tigermorning.github.io/korean-corrector-guide-bot/

실제 배포 주소에서 확인 완료(2026-08-28): 청크 로드, 검색, 스트리밍 답변,
출처 칩, 6필드 판정, 피드백 버튼 전부 정상 동작. CORS는 로컬 Ollama에
`OLLAMA_ORIGINS=https://tigermorning.github.io`를 설정하고 재시작해 확인했다.

**알려진 한계**: 로컬 소형 모델(`qwen3.5:2b`, 양자화)이 드물게 한글 대신
한자를 섞어 쓴다(예: "규칙집입니다"를 "규칙集입니다"로). 시스템·데이터
버그가 아니라 모델 자체의 생성 잡음이다 — 더 큰 모델로 바꾸면 줄어들
것으로 보이나 별도로 검증하지 않았다.

## 사용 조건 (숨기지 않고 미리 알림)

- **로컬 Ollama가 실행 중이어야 한다.** 서버가 없으므로 답변 생성·판정 둘 다
  이 조건에 걸린다.
- **첫 방문 시 임베딩 모델(약 200MB)을 내려받는다.** 브라우저 캐시에 한 번만
  저장되며(같은 브라우저 재방문 시 재다운로드 안 함), 첫 로드는 5~15초 걸릴 수
  있다.
- **배포 주소(GitHub Pages 등)에서 열었다면 CORS 허용이 필요하다.** 아래
  "Ollama 실행 + CORS 설정" 참고.
- **Chrome·Edge를 권장한다.** Safari는 Ollama 연결과 임베딩(WASM) 실행이
  불안정할 수 있다.

## 로컬에서 실행

```bash
git clone <이 저장소 URL>
cd korean-corrector-guide-bot
python -m http.server 8787   # 정적 파일이라 아무 정적 서버면 된다
```

브라우저에서 `http://localhost:8787` 접속. 빌드 단계 없음(순수 HTML/JS, ES 모듈).

## Ollama 실행 + CORS 설정

```bash
ollama serve
ollama pull qwen3.5:2b
```

페이지가 로컬(`http://localhost:8787`)이 아니라 GitHub Pages 같은 다른 origin에서
열렸다면, 그 origin이 로컬 Ollama(`localhost:11434`)를 호출할 수 있도록
`OLLAMA_ORIGINS`를 허용해야 한다. **정적 페이지를 배포했다고 서버가 로컬 모델을
대신 호출하는 게 아니다** — 배포된 페이지를 연 사용자의 브라우저가 여전히 그
사용자 자신의 `localhost:11434`를 직접 호출한다.

```bash
# macOS
launchctl setenv OLLAMA_ORIGINS "https://YOUR_ID.github.io"

# Windows (PowerShell, 관리자)
setx OLLAMA_ORIGINS "https://YOUR_ID.github.io"

# systemd 환경
systemctl set-environment OLLAMA_ORIGINS="https://YOUR_ID.github.io"
```

설정 후 Ollama를 재시작해야 반영된다. 화면 위 "엔진 설정"에서 서버 주소와 모델
이름을 확인/변경할 수 있다(기본값 `http://localhost:11434`, `qwen3.5:2b`).

## 설계 결정 요약

- **정적 앱, 서버 없음.** 검색(임베딩+BM25)은 브라우저에서, 생성·판정은 사용자의
  로컬 Ollama에서 — API 키 없이 전부 동작한다.
- **임베딩: `onnx-community/embeddinggemma-300m-ONNX`(768차원, dtype q4).**
  처음엔 `multilingual-e5-small`(384차원)로 만들었으나 관련/무관 질문의 코사인
  구간이 겹쳐 임계값을 못 잡았다 — 모델을 바꾸자 구간이 완전히 갈라졌다
  (`docs/EXPERIMENTS.md` 사이클 2).
- **하이브리드 검색**: 코사인 상위 10개 + 아직 안 들어온 청크 중 BM25 상위 5개
  (중복 제거). 벡터가 못 잡는 고유명사·정확한 표기를 BM25가 보충한다.
  hitRate 100%(평가셋 25문항 기준).
- **약한 근거(코사인 < 0.55) 처리**: 검색 결과를 버리지 않는다. 프롬프트를
  보수화하고 UI에 경고만 낸다 — 하드 차단은 임계값이 조금만 틀려도 답할 수
  있는 질문을 막아 버리는 문제가 실측으로 확인됐다(사이클 1).
- **판정(LLM-as-a-Judge) 6필드**: `grounded·noHalluc·cited·refusal·score·comment`.
  로컬 `qwen3.5:2b`로 답변 생성과 판정을 모두 하므로 독립 심사는 아니다 — 특히
  `refusal` 필드는 판정 모델의 자기 보고를 안 믿고 답변 텍스트를 직접 검사해
  코드가 재확정한다(작은 모델이 이 판단을 자주 틀리는 것이 확인됨).
- **인용 프롬프트 실험**: "표시하세요"에서 "각 문장 끝마다 [id]를 반드시 표시"로
  바꾸자 인용률 32%→45%, grounded 95%→100%, 평균 점수 78.0→82.5로 전부 올라
  채택(`docs/EXPERIMENTS.md` 사이클 3).
- **경쟁사 비교 질문은 프롬프트가 아니라 코드로 처리한다.** "다른 맞춤법
  검사기랑 차이가 뭐야?" 같은 질문에서, 프롬프트 지시만으로는 실제 존재하는
  서비스에 대한 거짓 정보(없는 기능·이름)를 반복해서 지어냈다. 지금은 그런
  질문이 감지되면 모델에 보내는 질문 자체를 "이 교정기 특징을 알려줘"로
  코드가 바꿔치고, 고정 안내 문구도 모델이 아니라 코드가 붙인다
  (`docs/EXPERIMENTS.md` 사이클 7).

## 실험 결과 요약

전체 실험 기록은 [docs/EXPERIMENTS.md](./docs/EXPERIMENTS.md). 재현:

```bash
npm install
npm run build:vectorstore   # data/vectorstore.json 재생성
npm run eval                # data/eval/results.json 생성 (Ollama 실행 중이어야 함)
```

| 실험 | 핵심 지표 | 결과 |
|---|---|---|
| 임베딩 모델 e5→EmbeddingGemma | 무관/관련 질문 코사인 겹침 | 있음 → 없음(0.19 여유) |
| 하이브리드 검색 vs 벡터 단독 | hitRate(25문항) | 0.95 → 1.00 |
| 약한 근거: 하드 차단 → 프롬프트 보수화 | 오차단 건수 | 3건 → 0건 |
| 인용 지시문 강화(A→B) | cited 비율 / 평균 점수 | 32%→45% / 78.0→82.5 |

## 저장소 구조

```
index.html, app.js, style.css   화면과 로직(빌드 없음, ES 모듈)
lib/                            검색(bm25/search)·Ollama 통신·프롬프트 (앱과 평가 스크립트가 공유)
data/chunks/*.json              사실 단위 청크 원본(사용법 5종 + 설계배경 4종)
data/vectorstore.json           청크 + 768차원 임베딩(정적 파일, 재생성 가능)
data/eval/                      평가셋(eval_set.jsonl)과 실행 결과(results.json)
scripts/                        벡터스토어 빌드, 평가, 임계값 보정 등 Node 스크립트
docs/                           PRD 보조 문서(설계원칙 발췌, 한계, 외래어 예외), 실험기록, 루브릭
```
