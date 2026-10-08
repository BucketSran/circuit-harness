/** Opt-in development pilot limits; no provider payloads or credentials are logged. */
import { appendFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  const maxCalls = Number(process.env.CHIPS_MAX_MODEL_CALLS);
  const maxBytes = Number(process.env.CHIPS_MAX_REQUEST_BYTES);
  const maxTokens = Number(process.env.CHIPS_MAX_OUTPUT_TOKENS);
  const trace = process.env.CHIPS_MODEL_BUDGET_TRACE;
  if (![maxCalls, maxBytes, maxTokens].every((x) => Number.isSafeInteger(x) && x > 0) || !trace) {
    throw new Error("Missing explicit model pilot limits");
  }
  let calls = 0;
  pi.on("context", (event) => {
    const remaining = Math.max(0, maxCalls - calls);
    const content = `Harness budget: request ${calls + 1} of ${maxCalls}; ` +
      `remaining including this request: ${remaining}. ` +
      `Maximum output per request: ${maxTokens} tokens (including reasoning where charged). ` +
      `Server tool replies report separate action/simulation budgets. ` +
      (remaining <= 2 ? "Finish your candidate and submit now if possible; no extra requests are reserved beyond this limit." : "Plan to submit within this budget.");
    appendFileSync(trace, JSON.stringify({ event: "budget_context", at: Date.now() / 1000,
      call: calls + 1, content }) + "\n", { mode: 0o600 });
    return { messages: [...event.messages, { role: "user", content, timestamp: Date.now() }] };
  });
  pi.on("message_end", (event) => {
    const message = event.message as { role?: string; usage?: unknown; stopReason?: string };
    if (message.role === "assistant") {
      appendFileSync(trace, JSON.stringify({ event: "response", at: Date.now() / 1000,
        usage: message.usage ?? null,
        stopReason: message.stopReason ?? null }) + "\n", { mode: 0o600 });
    }
  });
  pi.on("before_provider_request", (event) => {
    const payload = { ...(event.payload as Record<string, unknown>) };
    if ("max_completion_tokens" in payload) payload.max_completion_tokens = maxTokens;
    else payload.max_tokens = maxTokens;
    const bytes = Buffer.byteLength(JSON.stringify(payload));
    if (calls >= maxCalls || bytes > maxBytes) {
      appendFileSync(trace, JSON.stringify({ event: "budget_stop", at: Date.now() / 1000,
        calls, bytes, reason: calls >= maxCalls ? "model_request_limit" : "request_bytes_limit" }) + "\n", { mode: 0o600 });
      // Pi reports extension exceptions and continues the request. Terminate this
      // dedicated pilot process synchronously before the provider sees it.
      process.exit(73);
    }
    calls += 1;
    appendFileSync(trace, JSON.stringify({ event: "request", at: Date.now() / 1000,
      call: calls, bytes, maxTokens }) + "\n", { mode: 0o600 });
    return payload;
  });
}
