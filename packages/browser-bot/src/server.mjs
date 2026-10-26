// The MCP server itself: a low-level server with plain JSON Schema tools.
//
// The low-level server is used on purpose. Tool schemas travel to the client verbatim, so
// the bot's contract does not depend on the SDK's schema library, and an agent sees exactly
// the descriptions written in tools.mjs.

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { JobStore } from "./jobs.mjs";
import { asToolResult, failure, makeHandlers, tools } from "./tools.mjs";
import { BotError } from "./values.mjs";

export const VERSION = "0.1.0";

const INSTRUCTIONS = `This server does browser tasks on any website, deterministically.

Start with browser_doctor when something fails: it names the missing piece (Node, Chromium, the
Pomerado command, the model key).

The normal loop is one call: browser_do with a URL and the task in plain language. The first call
for a site mints an integration, which explores the site in a real browser and takes minutes; it
stops and asks questions when it needs an account, a one-time code or a choice. Answer those with
provide_input, and follow long work with get_job. Later calls for the same site replay the saved
integration, with no model and usually in seconds.

Choose authority "read" unless the task changes something. "write" needs approved_by_user: true,
set only after your user has agreed to the change in this conversation, and a write result says
whether the change was confirmed.

Never send a password or a one-time code through provide_input without asking your user first:
answers are visible to this conversation, and nothing is saved.`;

/**
 * Build the server and its job store. Nothing starts a browser or calls a model until a tool
 * that needs one is invoked.
 */
export function createBotServer({ config, env = process.env, version = VERSION }) {
  const jobs = new JobStore({ maxJobs: config.maxJobs });
  const handlers = makeHandlers({ config, jobs, env });
  const server = new Server(
    { name: "browser-bot", version },
    {
      capabilities: { tools: {} },
      instructions: INSTRUCTIONS,
    },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: tools.map((tool) => ({
      name: tool.name,
      title: tool.title,
      description: tool.description,
      inputSchema: tool.inputSchema,
      annotations: tool.annotations,
    })),
  }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const name = request.params?.name;
    const handler = handlers[name];
    if (handler === undefined) {
      return failure(new BotError("unknown_tool", `Unknown tool "${String(name)}".`));
    }
    try {
      const payload = await handler(request.params?.arguments ?? {});
      const result = asToolResult(payload);
      // A job that already failed inside the wait is an error result, not a success: the
      // agent must not treat "nothing ran" as an answer.
      if (payload !== null && typeof payload === "object" && payload.status === "failed") {
        result.isError = true;
      }
      return result;
    } catch (error) {
      return failure(error);
    }
  });

  return { server, jobs, handlers, config };
}

/** Serve the bot on stdio, the transport every MCP client speaks. */
export async function serveStdio({ config, env = process.env, version = VERSION } = {}) {
  if (config === undefined) throw new Error("serveStdio needs a config");
  const bot = createBotServer({ config, env, version });
  const transport = new StdioServerTransport();
  let closing = false;
  const shutdown = async (code = 0) => {
    if (closing) return;
    closing = true;
    await bot.jobs.cancelAll().catch(() => {});
    await bot.server.close().catch(() => {});
    await transport.close().catch(() => {});
    process.exitCode = code;
  };
  process.once("SIGINT", () => void shutdown(0));
  process.once("SIGTERM", () => void shutdown(0));
  transport.onclose = () => void shutdown(0);
  await bot.server.connect(transport);
  return bot;
}
