// A supervised MCP child process: the Pomerado minter, or one integration's server.
//
// Each child speaks MCP over stdio. The bot is a client of them and never parses their
// standard output itself, so a child's logs cannot corrupt the bot's own protocol stream.

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { BotError, parseToolResult } from "./values.mjs";

const STDERR_TAIL = 40;

export class PomeradoChild {
  #client;
  #transport;
  #stderr = [];

  constructor({ client, transport, label }) {
    this.#client = client;
    this.#transport = transport;
    this.label = label;
    this.closed = false;
  }

  /**
   * Start a child and complete the MCP handshake.
   *
   * `command` is an argv array, already configured. The child inherits the bot's
   * environment, because the minter reads OPENAI_API_KEY from it: the key never travels
   * through a tool argument or a chat message.
   */
  static async connect({ command, env, cwd, label, timeoutMs = 90_000 }) {
    const [file, ...args] = command;
    if (file === undefined) throw new BotError("bad_command", "No command to start.");
    const transport = new StdioClientTransport({
      command: file,
      args,
      env,
      cwd,
      stderr: "pipe",
    });
    const client = new Client({ name: "browser-bot", version: "0.1.0" }, { capabilities: {} });
    const child = new PomeradoChild({ client, transport, label });
    transport.stderr?.on("data", (chunk) => {
      const text = String(chunk);
      for (const line of text.split("\n")) {
        if (line.trim() === "") continue;
        child.#stderr.push(line);
        if (child.#stderr.length > STDERR_TAIL) child.#stderr.shift();
      }
    });
    try {
      await client.connect(transport);
    } catch (cause) {
      await child.close().catch(() => {});
      throw new BotError(
        "child_unavailable",
        `Could not start ${label} (${command.join(" ")}): ${child.stderrTail() || cause?.message || cause}. ` +
          "Check that Pomerado is installed and that BROWSER_BOT_POMERADO points at pomerado-mcp.",
        cause,
      );
    }
    // A successful handshake can still be a process that dies on its first call; a quick
    // tools/list is what proves the child is really a usable MCP server.
    try {
      await client.listTools(undefined, { timeout: timeoutMs });
    } catch (cause) {
      await child.close().catch(() => {});
      throw new BotError(
        "child_unavailable",
        `${label} started but did not answer tools/list: ${child.stderrTail() || cause?.message || cause}`,
        cause,
      );
    }
    return child;
  }

  stderrTail(limit = 6) {
    return this.#stderr.slice(-limit).join(" | ");
  }

  async listTools() {
    const result = await this.#client.listTools(undefined, { timeout: 60_000 });
    return result?.tools ?? [];
  }

  /**
   * Call a tool and return its decoded output.
   *
   * Pomerado answers its helper tools with a job view, and its own tools with either the
   * operation output or a job id; both arrive as JSON, so the caller decides what it got.
   * A tool that reports isError is turned into a thrown BotError, because a failed website
   * action must never look like a success.
   */
  async call(name, args, { timeoutMs = 120_000 } = {}) {
    if (this.closed) throw new BotError("child_closed", `${this.label} is already closed.`);
    let result;
    try {
      result = await this.#client.callTool({ name, arguments: args }, undefined, {
        timeout: timeoutMs,
        resetTimeoutOnProgress: true,
      });
    } catch (cause) {
      throw new BotError(
        "child_call_failed",
        `${this.label} tool "${name}" failed: ${cause?.message ?? cause}`,
        cause,
      );
    }
    if (result?.isError === true) {
      const text = Array.isArray(result.content)
        ? result.content.filter((part) => part?.type === "text").map((part) => part.text).join(" ").trim()
        : "";
      throw new BotError(
        "child_tool_error",
        text === "" ? `${this.label} tool "${name}" failed.` : text,
      );
    }
    return parseToolResult(result);
  }

  async close() {
    if (this.closed) return;
    this.closed = true;
    try {
      await this.#client.close();
    } finally {
      try {
        await this.#transport.close();
      } catch {
        /* already gone */
      }
    }
  }
}
