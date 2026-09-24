// claude-code-router custom router that asks your local handoff service where each request should go.
//
// Setup (claude-code-router ≥ v3):
//   1. copy this file into your CCR config directory, e.g. ~/.claude-code-router/handoff-router.js
//   2. in config.json:  "CUSTOM_ROUTER_PATH": "~/.claude-code-router/handoff-router.js"
//   3. set CHEAP_MODEL below to a "provider,model" you have configured (e.g. "ollama,qwen3:8b")
//   4. warm it up once: `handoff route "hello"` (the first call loads the model, ~50 s)
//
// Contract: returns CHEAP_MODEL only when handoff says "cheap". Strong, errors, timeouts or a cold service all
// return undefined, so CCR's own rules decide — handoff can only ever save money, never block a request.

const fs = require("fs");
const os = require("os");
const path = require("path");

const CHEAP_MODEL = process.env.HANDOFF_CHEAP_MODEL || "ollama,qwen3:8b";
const URL = `http://127.0.0.1:${process.env.HANDOFF_PORT || 7071}/route`;
const TIMEOUT_MS = 1500;   // first call from Node can take ~0.4 s; a cold service never blocks CCR for longer
const TOKEN_FILE = path.join(process.env.HANDOFF_HOME || path.join(os.homedir(), ".handoff"), "token");

function lastUserText(body) {
  const messages = (body && body.messages) || [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== "user") continue;
    if (typeof m.content === "string") return m.content;
    if (Array.isArray(m.content)) {
      const text = m.content.filter((p) => p.type === "text").map((p) => p.text).join(" ");
      if (text.trim()) return text;
    }
  }
  return "";
}

module.exports = async function handoffRouter(req) {
  try {
    const request = lastUserText(req.body).slice(0, 4000);
    if (!request.trim()) return undefined;
    const token = fs.readFileSync(TOKEN_FILE, "utf8").trim();
    const res = await fetch(URL, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Handoff-Token": token },
      body: JSON.stringify({ request }),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    const out = await res.json();
    return out.route === "cheap" ? CHEAP_MODEL : undefined;
  } catch {
    return undefined; // fail open: let claude-code-router's own rules decide
  }
};
