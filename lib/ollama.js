// 로컬 Ollama와 통신하는 세 경로.
// 1) streamChat — /api/chat, stream:true. 답변 생성용. NDJSON을 한 줄씩
//    파싱해 델타 텍스트를 yield한다. 취소는 호출자의 AbortController.signal로.
// 2) chatJSON — /api/chat, stream:false + format:"json" + temperature:0.
//    판정(LLM-as-a-Judge) 전용. 스트리밍하지 않고 완성된 JSON 문자열 하나를
//    그대로 받는다(중간에 파싱할 것이 없다).
// 3) checkHealth — /api/tags. 채팅 시작 전 엔진이 켜져 있는지 확인해,
//    "모델이 꺼짐"과 "생성 실패"를 구분한 배너를 보여줄 수 있게 한다.

export async function* streamChat({ baseUrl, model, messages, signal }) {
  const res = await fetch(`${baseUrl.replace(/\/$/, "")}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // think:false — qwen3 계열처럼 사고 모드가 있는 모델은 꺼 둔다. 켜 두면
    // content가 빈 채로 한참 사고 토큰만 스트리밍되어 화면이 비어 보이고,
    // 작은 모델은 사고에 예산을 다 써서 실제 답을 못 낼 때도 있었다(실측).
    body: JSON.stringify({ model, messages, stream: true, think: false }),
    signal,
  });
  if (!res.ok || !res.body) {
    const bodyText = await res.text().catch(() => "");
    throw new Error(`Ollama 응답 실패 (${res.status}): ${bodyText || res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const json = JSON.parse(trimmed);
      if (json.message?.content) yield json.message.content;
      if (json.done) return;
    }
  }
}

/** 스트리밍 없이 한 번에 답을 받는다(호환용 — 내부적으로 streamChat을 소진). */
export async function chatOnce({ baseUrl, model, messages, signal }) {
  let full = "";
  for await (const delta of streamChat({ baseUrl, model, messages, signal })) {
    full += delta;
  }
  return full;
}

/** 판정 전용: stream:false + format:"json" + temperature:0. 완성 JSON 문자열 하나. */
export async function chatJSON({ baseUrl, model, messages, signal }) {
  const res = await fetch(`${baseUrl.replace(/\/$/, "")}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      think: false,
      format: "json",
      options: { temperature: 0 },
    }),
    signal,
  });
  if (!res.ok) {
    const bodyText = await res.text().catch(() => "");
    throw new Error(`Ollama 판정 호출 실패 (${res.status}): ${bodyText || res.statusText}`);
  }
  const json = await res.json();
  return json.message?.content ?? "";
}

/** 채팅 시작 전 엔진이 켜져 있는지 확인한다. 생성 실패와 "엔진 꺼짐"을 구분하는 용도. */
export async function checkHealth(baseUrl) {
  try {
    const res = await fetch(`${baseUrl.replace(/\/$/, "")}/api/tags`, { method: "GET" });
    if (!res.ok) return { ok: false, reason: `HTTP ${res.status}` };
    return { ok: true };
  } catch (err) {
    return { ok: false, reason: err.message };
  }
}

/** 응답 텍스트에서 첫 JSON 객체를 관대하게 뽑아낸다(소형 모델은 여분의 말을 덧붙이곤 함). */
export function extractFirstJson(text) {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1 || end < start) return null;
  try {
    return JSON.parse(text.slice(start, end + 1));
  } catch {
    return null;
  }
}
