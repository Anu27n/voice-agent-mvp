// Shared by the call page and the history page: API calls that carry the
// deployment's access code (when one is configured) and small formatters.

const CODE_KEY = "voice-agent-access-code";

export function accessCode() {
  try {
    return sessionStorage.getItem(CODE_KEY) || "";
  } catch {
    return "";
  }
}

function rememberCode(code) {
  try {
    sessionStorage.setItem(CODE_KEY, code);
  } catch {
    // storage unavailable (private mode): the prompt will appear again next time
  }
}

// fetch() that adds the access code and, on 401, asks for it once and retries.
export async function api(path, options = {}) {
  for (let attempt = 0; ; attempt++) {
    const headers = new Headers(options.headers || {});
    const code = accessCode();
    if (code) headers.set("X-Access-Code", code);
    const res = await fetch(path, { ...options, headers });
    if (res.status !== 401 || attempt >= 2) return res;
    const entered = window.prompt(
      attempt === 0 ? "This voice agent is protected. Enter the access code:" : "That code didn't work. Try again:",
    );
    if (!entered) return res;
    rememberCode(entered.trim());
  }
}

// Adds the access code to a URL that can't carry headers (WebSocket, download links).
export function withCode(url) {
  const code = accessCode();
  if (!code) return url;
  return `${url}${url.includes("?") ? "&" : "?"}code=${encodeURIComponent(code)}`;
}

export function formatMs(ms) {
  return ms == null ? "-" : `${Math.round(ms)} ms`;
}
