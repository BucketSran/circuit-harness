// Copyright 2026 TMLR Group
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Give pi the tools an MCP server offers, since pi has no MCP client.
//
// The client lives here instead, talking to the same server Claude Code and
// Codex spawn: MCP stdio is newline-delimited JSON-RPC and needs four methods,
// which is far less code than a second server would be.
//
// Tools register under their plain names, which is why bridging pi is worth
// doing at all: its seven built-in ids are exactly this repository's
// PUBLIC_TOOL_IDS, so the model sees `bash`, not `mcp__alphaapollo__bash`.
//
// Server declarations arrive as JSON in ALPHAAPOLLO_MCP_SERVERS -- the same
// `mcpServers` shape the other agents take, written by PiSession, because pi
// cannot pass arguments to an extension. Stdio only.

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn, type ChildProcess } from "node:child_process";
import { writeSync } from "node:fs";

const ENV_KEY = "ALPHAAPOLLO_MCP_SERVERS";

interface StdioServer {
  command: string;
  args?: string[];
  env?: Record<string, string>;
}

interface TextBlock {
  type: string;
  text?: string;
}

/** One MCP server over stdio: spawn, request, close. */
class McpStdioClient {
  private child: ChildProcess;
  private buffer = "";
  private nextId = 1;
  private pending = new Map<number, (message: any) => void>();

  constructor(
    readonly label: string,
    server: StdioServer,
  ) {
    this.child = spawn(server.command, server.args ?? [], {
      stdio: ["pipe", "pipe", "inherit"],
      env: { ...process.env, ...(server.env ?? {}) },
    });
    this.child.stdout!.setEncoding("utf8");
    this.child.stdout!.on("data", (chunk: string) => {
      this.buffer += chunk;
      for (let cut = this.buffer.indexOf("\n"); cut >= 0; cut = this.buffer.indexOf("\n")) {
        const line = this.buffer.slice(0, cut).trim();
        this.buffer = this.buffer.slice(cut + 1);
        if (line) this.deliver(JSON.parse(line));
      }
    });
    // Without this a dead server leaves every call hanging until pi's timeout.
    this.child.on("exit", (code) => {
      for (const waiter of this.pending.values())
        waiter({ error: { message: `server exited (${code})` } });
      this.pending.clear();
    });
  }

  private deliver(message: any): void {
    // Server-initiated requests need no answer: this client advertises no
    // capabilities, so nothing should be asking it for one.
    const waiter = this.pending.get(message.id);
    if (!waiter) return;
    this.pending.delete(message.id);
    waiter(message);
  }

  private request(method: string, params: Record<string, unknown> = {}): Promise<any> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, (message) =>
        message.error
          ? reject(new Error(`${this.label}: ${message.error.message ?? "MCP error"}`))
          : resolve(message.result),
      );
      this.child.stdin!.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`);
    });
  }

  async initialize(): Promise<void> {
    await this.request("initialize", {
      protocolVersion: "2025-06-18",
      capabilities: {},
      clientInfo: { name: "alphaapollo-pi-extension", version: "1" },
    });
    this.child.stdin!.write(
      `${JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" })}\n`,
    );
  }

  async listTools(): Promise<
    Array<{ name: string; description?: string; inputSchema: Record<string, unknown> }>
  > {
    return (await this.request("tools/list"))?.tools ?? [];
  }

  callTool(name: string, args: unknown): Promise<{ content?: TextBlock[]; isError?: boolean }> {
    return this.request("tools/call", { name, arguments: args ?? {} });
  }

  close(): void {
    this.child.stdin!.end();
    this.child.kill();
  }
}

export default async function (pi: ExtensionAPI) {
  const declared = JSON.parse(process.env[ENV_KEY] ?? "{}") as Record<string, StdioServer>;
  const clients: McpStdioClient[] = [];
  const owners = new Map<string, string>();

  for (const [label, server] of Object.entries(declared)) {
    const client = new McpStdioClient(label, server);
    clients.push(client);
    await client.initialize();

    for (const tool of await client.listTools()) {
      const previous = owners.get(tool.name);
      if (previous !== undefined) {
        throw new Error(`tool ${tool.name} is offered by both ${previous} and ${label}`);
      }
      owners.set(tool.name, label);
      // Operator JSONL evidence only, never a model message. Pi redirects
      // extension stdout.write; writeSync preserves these protocol records.
      writeSync(1, `${JSON.stringify({ type: "alphaapollo_mcp_tool", toolName: tool.name })}\n`);
      pi.registerTool({
        name: tool.name,
        label: tool.name,
        description: tool.description ?? tool.name,
        parameters: tool.inputSchema as any,
        async execute(toolCallId: string, params: unknown) {
          writeSync(1, `${JSON.stringify({ type: "alphaapollo_mcp_dispatch", toolCallId, toolName: tool.name })}\n`);
          const result = await client.callTool(tool.name, params);
          const content = (result.content ?? []).filter((block) => block.type === "text");
          const text = content.map((block) => block.text ?? "").join("\n");
          // pi marks a result failed only when execute throws, and the text it
          // carries is the same payload a successful call would have returned.
          if (result.isError) throw new Error(text);
          return { content: content as any, details: {} };
        },
      });
    }
  }

  pi.on("session_shutdown", async () => {
    for (const client of clients) client.close();
  });
}
