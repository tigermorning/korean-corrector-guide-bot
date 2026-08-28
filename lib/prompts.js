// app.js(브라우저)와 scripts/run_eval.mjs(Node 평가 스크립트)가 똑같은 프롬프트를
// 쓰도록 공용으로 뺐다. 여기를 안 거치고 각자 프롬프트 문자열을 따로 관리하면
// 평가 결과가 실제 서비스 동작과 어긋나게 된다.

export const REFUSAL_MARKER = "확인되지";

// 프롬프트 지시만으로는 소형 모델이 다른 서비스에 대한 구체적 주장(존재하지
// 않는 기능·이름)을 지어내는 것을 못 막는다는 게 실측으로 두 번 확인됐다
// (docs/EXPERIMENTS.md 사이클 7) — 실제 서비스에 대한 거짓 정보라 위험도가
// 더 크므로, 프롬프트가 아니라 코드로 원천 차단한다: 질문에서 이 감지에
// 걸리면 모델에게 넘기는 질문 자체에서 이름을 지우고, 답변에서도 한 번 더
// 걸러낸다(아래 sanitizeCompetitorMentions·stripCompetitorSentences).
const COMPETITOR_TRIGGER = /부산대|네이버\s*맞춤법|다음\s*맞춤법|한컴|아래아\s*한글|다른\s*(맞춤법|교정)\s*(검사기|프로그램|도구|서비스)/;

export function mentionsCompetitor(text) {
  return COMPETITOR_TRIGGER.test(text);
}

/** 모델에게 보낼 질문에서 경쟁사 언급을 지우고 "이 교정기 자체 설명"만
 * 요청하는 질문으로 바꾼다 — 모델 컨텍스트에 이름이 아예 없으면 그 이름을
 * 화제로 삼을 트리거도 사라진다(사전학습 지식 회상까지 100% 막지는 못하지만
 * 크게 줄어든다는 것을 실측으로 확인했다). */
export function sanitizeCompetitorQuestion(question) {
  if (!mentionsCompetitor(question)) return question;
  return "이 교정기가 무엇을 어떻게 하는지 특징을 알려줘.";
}

/** 그래도 답변에 경쟁사 언급이 남아 있으면(사전학습 지식 회상 등) 그
 * 문장만 제거하는 마지막 안전망. 문장 전체를 지우는 거친 방법이지만,
 * 실제 서비스에 대한 근거 없는 주장을 화면에 내보내는 것보다 안전하다. */
export function stripCompetitorSentences(answer) {
  const sentences = answer.split(/(?<=[.!?\n])/);
  return sentences.filter((s) => !COMPETITOR_TRIGGER.test(s)).join("").trim();
}

/** 경쟁사 질문일 때 화면에 항상 붙는 고정 안내문 — 모델이 만들지 않고
 * 코드가 그대로 붙인다(모델 생성에 맡기면 이 문장 자체도 흔들린다). */
export const COMPETITOR_DISCLAIMER =
  "다른 맞춤법 검사기와 비교한 자료는 이 챗봇에 없습니다. 이 교정기 자체가 하는 일은 다음과 같습니다.\n\n";

/** 상대 시간 표현("지금"·"오늘"·"이번 주")을 해석할 기준 시각. KST 고정. */
export function nowKST() {
  const fmt = new Intl.DateTimeFormat("ko-KR", {
    timeZone: "Asia/Seoul",
    dateStyle: "full",
    timeStyle: "short",
  });
  return `${fmt.format(new Date())} KST`;
}

/**
 * @param {Array} sources hybridSearch() 결과
 * @param {{weak?: boolean}} opts weak=true면 근거가 약하다는 것을 명시하고
 *   프롬프트를 보수적으로 바꾼다 — 검색 결과를 버리지 않고 그대로 넘기되
 *   모델이 단정 대신 자료의 한계를 말하게 하는 것이 핵심이다.
 */
export function buildSystemPrompt(sources, opts = {}) {
  const context = sources.map((s) => `[${s.id} | ${s.section}]\n${s.text}`).join("\n\n");
  const lines = [
    "다음 자료는 한국어 교정기(korean-subtitle-corrector)의 사용법과 설계 배경을 다루는",
    "공개 문서에서 뽑은 조각입니다. 이 자료 안의 내용만 근거로 답하고,",
    "자료에 없는 내용은 절대 만들어내지 마세요.",
    "",
    "**코드·명령어·API·파일 경로·포트 번호는 절대 새로 지어내지 마세요.**",
    "근거 조각에 정확히 그 문자열이 있을 때만 그대로 옮겨 적으세요.",
    "근거에 실행 방법이 여러 개(웹 화면, 명령줄 등) 있으면, 컴퓨터 사용이 서툰",
    "사람도 따라 할 수 있는 방법(웹 화면)을 먼저 안내하세요.",
    "'API'·'서버'·'엔드포인트' 같은 개발자 용어는 근거에 그 단어가 있어도",
    "일반 사용자 설명에서는 쓰지 마세요 — '한 번만 켜 두는 것'·'그 다음부터는",
    "웹페이지만 열면 됨'처럼 처음 한 번 하는 일과 매번 하는 일을 구분해 말하세요.",
    "",
    "불편해 보이는 점을 답할 때(예: 붙여넣기 칸이 없음)는 사과하는 투로 끝내지",
    "말고, 근거에 있다면 그 방식이 왜 그런지(예: 자막 서식 보존)도 함께 알려",
    "주세요. 다른 서비스 이름을 언급하거나 비교·비판하지는 마세요 — 근거에",
    "없는 다른 서비스에 대한 주장은 절대 지어내지 마세요(실제 존재하는",
    "서비스에 대한 거짓 정보라 특히 위험합니다). 이 교정기 자체의 사실만",
    "말하세요.",
  ];
  if (opts.weak) {
    lines.push(
      "",
      "주의: 검색된 조각의 유사도가 낮습니다. 질문과 완전히 맞는 근거가 아닐 수 있으니,",
      "근거에 있는 내용만 짧게 답하고 자료에 없는 부분은 없다고 말합니다.",
      "자료가 질문과 전혀 무관하면 '문서에서 확인되지 않습니다'라고 솔직히 답하세요."
    );
  } else {
    lines.push("", "근거로 답할 수 없으면 '문서에서 확인되지 않습니다'라고 솔직히 답하세요.");
  }
  lines.push(
    "",
    "**중요 — 인용 규칙**: 답변의 각 문장 또는 각 항목 끝에 그 내용의 근거가 된",
    "조각의 [id]를 반드시 표시하세요. 예: '자막 모드에서는 편집 관례가 적용됩니다 [usage-index-doctype-default].'",
    "[id] 표시가 없는 문장은 근거 없는 주장으로 간주되니 절대 빠뜨리지 마세요.",
    "인용은 반드시 대괄호 텍스트 [id] 하나로만 쓰세요. [id](경로)처럼",
    "마크다운 링크로 만들지 말고, 파일 경로·URL·file:// 주소를 답변 안에",
    "절대 새로 지어내지 마세요 — 실제 원문 링크는 화면의 출처 칩이 따로 보여줍니다.",
    `현재 시각은 ${nowKST()}입니다. '지금', '오늘', '이번 주' 같은 상대 표현은 이 시각을 기준으로 해석합니다.`,
    "한국어로, 간결하게 답하세요.",
    "",
    "[자료]",
    context
  );
  return lines.join("\n");
}

/**
 * LLM-as-a-Judge 프롬프트. 6개 필드만 JSON으로 반환하게 해, 화면 쪽이 설명
 * 문장을 추측하지 않고 정해진 필드를 배지로 바로 연결할 수 있게 한다.
 */
export function buildJudgePrompt({ question, answer, sources }) {
  const context = sources.map((s) => `[${s.id}] ${s.text}`).join("\n");
  return [
    "당신은 RAG 챗봇 답변의 평가자입니다. 아래 [질문], [근거자료], [답변]을 읽고",
    "다음 기준으로 JSON만 출력합니다.",
    "grounded: 답변 내용이 근거자료에서 나왔는가 (true/false)",
    "noHalluc: 근거에 없는 사실을 지어내지 않았는가 (true/false)",
    "cited: 답변 안에 근거 조각의 [id] 표시가 있는가 (true/false)",
    "refusal: 근거에 답이 없어서 '없다'고 답한 경우 true, 그 외 false",
    "score: 0-100 정수 (grounded·noHalluc·cited 반영)",
    "comment: 한두 문장 평어 (한국어)",
    '출력 형식: {"grounded":bool,"noHalluc":bool,"cited":bool,"refusal":bool,"score":int,"comment":"..."} — JSON 외 텍스트 금지.',
    "",
    `[질문] ${question}`,
    `[근거자료]\n${context}`,
    `[답변] ${answer}`,
  ].join("\n");
}

/** 모델이 가끔 5점 만점처럼 score를 반환하는 것을 100점 척도로 보정한다. */
export function normalizeJudgeScore(score) {
  if (typeof score !== "number") return score;
  return score <= 5 ? Math.round((score / 5) * 100) : score;
}
