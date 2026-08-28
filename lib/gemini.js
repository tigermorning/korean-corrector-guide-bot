// Gemini API(REST, API 키 방식)와 통신 — lib/ollama.js와 같은 3가지 인터페이스를
// 맞춰서 app.js가 엔진만 바꿔 끼울 수 있게 한다. 브라우저에서 키를 직접 써서
// Google 엔드포인트를 호출하는 구조라, 키가 네트워크 탭에 노출된다는 한계를
// README/화면에 명시해야 한다(정적 사이트라 서버가 대신 숨겨줄 수 없음).

const BASE = "https://generativelanguage.googleapis.com/v1beta";

// 우리 messages 형식({role:"system"|"user"|"assistant", content})을
// Gemini 형식({role:"user"|"model", parts:[{text}]} + 별도 systemInstruction)으로 바꾼다.
function toGeminiRequest(messages) {
  const systemMsgs = messages.filter((m) => m.role === "system").map((m) => m.content);
  const contents = messages
    .filter((m) => m.role !== "system")
    .map((m) => ({ role: m.role === "assistant" ? "model" : "user", parts: [{ text: m.content }] }));
  const body = { contents };
  if (systemMsgs.length) body.systemInstruction = { parts: [{ text: systemMsgs.join("\n\n") }] };
  return body;
}

export async function* streamChat({ apiKey, model, messages, signal }) {
  const body = toGeminiRequest(messages);
  const res = await fetch(`${BASE}/models/${model}:streamGenerateContent?alt=sse&key=${encodeURIComponent(apiKey)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    const bodyText = await res.text().catch(() => "");
    throw new Error(`Gemini 응답 실패 (${res.status}): ${bodyText || res.statusText}`);
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
      if (!trimmed.startsWith("data:")) continue;
      const jsonStr = trimmed.slice(5).trim();
      if (!jsonStr || jsonStr === "[DONE]") continue;
      const json = JSON.parse(jsonStr);
      const text = json.candidates?.[0]?.content?.parts?.map((p) => p.text ?? "").join("");
      if (text) yield text;
    }
  }
}

export async function chatOnce({ apiKey, model, messages, signal }) {
  let full = "";
  for await (const delta of streamChat({ apiKey, model, messages, signal })) full += delta;
  return full;
}

/** 판정 전용: JSON 응답 강제 + temperature 0. */
export async function chatJSON({ apiKey, model, messages, signal }) {
  const body = toGeminiRequest(messages);
  body.generationConfig = { responseMimeType: "application/json", temperature: 0 };
  const res = await fetch(`${BASE}/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    const bodyText = await res.text().catch(() => "");
    throw new Error(`Gemini 판정 호출 실패 (${res.status}): ${bodyText || res.statusText}`);
  }
  const json = await res.json();
  return json.candidates?.[0]?.content?.parts?.map((p) => p.text ?? "").join("") ?? "";
}

/** 키가 유효한지 가볍게 확인한다(모델 목록 조회). */
export async function checkHealth(apiKey) {
  if (!apiKey) return { ok: false, reason: "API 키 없음" };
  try {
    const res = await fetch(`${BASE}/models?key=${encodeURIComponent(apiKey)}`);
    if (!res.ok) {
      const bodyText = await res.text().catch(() => "");
      return { ok: false, reason: `HTTP ${res.status}: ${bodyText.slice(0, 200)}` };
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, reason: err.message };
  }
}
