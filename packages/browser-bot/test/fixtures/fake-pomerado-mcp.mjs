#!/usr/bin/env node
// A stand-in for `pomerado-mcp`, used by the tests. It speaks the same MCP protocol and the
// same job vocabulary as the real server: `mint` answers with a job id, the job stops to ask
// a question, and completing it writes an integration folder to disk. `serve` mode loads such
// a folder and answers as the integration's own tool.
//
// Nothing here touches a browser, a model or the network, so the tests exercise the bot's
// orchestration, not Pomerado's.

import { mkdir, writeFile, readFile } from "node:fs/promises";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

const argv = process.argv.slice(2);
if (argv.includes("--help")) {
  process.stdout.write("fake-pomerado-mcp mint --root DIR | serve --artifact DIR\n");
  process.exit(0);
}
const mode = argv[0];
const flag = (name) => {
  const index = argv.indexOf(name);
  return index === -1 ? undefined : argv[index + 1];
};
const root = flag("--root");
const artifact = flag("--artifact");

const answerable = new Set(["yes", "no"]);

function text(value) {
  return {
    content: [{ type: "text", text: JSON.stringify(value) }],
    structuredContent: value,
  };
}

/** The mint job's state machine: running -> input_required -> completed. */
function makeMintJob({ name, url, intent, effect, onComplete }) {
  const job = {
    id: `job_${randomUUID()}`,
    status: "running",
    ticks: 0,
    answered: false,
    requestId: randomUUID(),
    output: undefined,
  };
  job.get = async () => {
    job.ticks += 1;
    if (job.ticks === 1) {
      return {
        job_id: job.id,
        status: "running",
        next: { tool: "get_job", arguments: { job_id: job.id } },
      };
    }
    if (!job.answered) {
      return {
        job_id: job.id,
        status: "input_required",
        pending_input: {
          id: job.requestId,
          source: "agent",
          notice: `The site asks whether to use the demo account on ${new URL(url).host}.`,
          questions: [
            {
              id: "account",
              type: "choice",
              prompt: "Which account should the integration use?",
              options: [
                { id: "yes", label: "The demo account" },
                { id: "no", label: "No account" },
              ],
              allowOther: false,
            },
          ],
          expires_at: new Date(Date.now() + 600_000).toISOString(),
        },
        next: {
          tool: "provide_input",
          arguments: { job_id: job.id, request_id: job.requestId },
        },
      };
    }
    if (job.output === undefined) job.output = await onComplete();
    return { job_id: job.id, status: "completed", output: job.output };
  };
  job.provide = (requestId, answers) => {
    if (requestId !== job.requestId) throw new Error("stale input request");
    if (!answerable.has(answers?.account)) throw new Error("invalid answers");
    job.answered = true;
    return { job_id: job.id, status: "running" };
  };
  return job;
}

async function mintIntegration({ name, url, intent, effect }) {
  const directory = join(root, name);
  await mkdir(join(directory, "src"), { recursive: true });
  const countPath = join(root, ".mint-count");
  const previous = Number.parseInt(await readFile(countPath, "utf8").catch(() => "0"), 10);
  await writeFile(countPath, String(previous + 1));
  await writeFile(
    join(directory, "deployment.json"),
    `${JSON.stringify(
      {
        name,
        description: `${effect} ${intent}`,
        request: { url, intent, effect },
      },
      null,
      2,
    )}\n`,
  );
  await writeFile(
    join(directory, "pomerado.json"),
    `${JSON.stringify(
      {
        entrypoint: "src/operation.mjs",
        inputSchema: {
          type: "object",
          properties: { query: { type: "string" } },
          additionalProperties: false,
        },
        outputSchema: {
          type: "object",
          properties: { heading: { type: "string" } },
        },
        questions: [],
      },
      null,
      2,
    )}\n`,
  );
  await writeFile(
    join(directory, "mcp.json"),
    `${JSON.stringify(
      {
        mcpServers: {
          [name]: {
            command: process.execPath,
            args: [new URL(import.meta.url).pathname, "serve", "--artifact", directory],
          },
        },
      },
      null,
      2,
    )}\n`,
  );
  return {
    build: { steps: 1, summary: "fake build" },
    integration: {
      directory,
      configPath: join(directory, "mcp.json"),
      launcherPath: join(directory, "mcp.mjs"),
    },
  };
}

function mintTools() {
  let job;
  return {
    list: [
      { name: "mint", description: "Mint", inputSchema: { type: "object" } },
      { name: "get_job", description: "Check a job", inputSchema: { type: "object" } },
      { name: "provide_input", description: "Answer", inputSchema: { type: "object" } },
      { name: "cancel_job", description: "Cancel", inputSchema: { type: "object" } },
    ],
    call(name, args) {
      if (name === "mint") {
        if (job !== undefined) throw new Error("busy");
        job = makeMintJob({ ...args, onComplete: () => mintIntegration(args) });
        return {
          job_id: job.id,
          status: "running",
          next: { tool: "get_job", arguments: { job_id: job.id } },
        };
      }
      if (job === undefined) throw new Error("unknown job");
      if (name === "get_job") return job.get();
      if (name === "provide_input") return job.provide(args.request_id, args.answers);
      if (name === "cancel_job") {
        const cancelled = { job_id: job.id, status: "cancelled" };
        job = undefined;
        return cancelled;
      }
      throw new Error(`unknown tool ${name}`);
    },
  };
}

async function integrationTools(directory) {
  const deployment = JSON.parse(await readFile(join(directory, "deployment.json"), "utf8"));
  const pomerado = JSON.parse(await readFile(join(directory, "pomerado.json"), "utf8"));
  const name = deployment.name;
  const effect = deployment.request.effect;
  let job;

  const runOnce = (input) => ({
    heading: `heading from ${new URL(deployment.request.url).host}`,
    effect,
    input,
    write_status: effect === "write" ? "confirmed" : undefined,
  });

  return {
    list: [
      { name, description: deployment.description, inputSchema: pomerado.inputSchema },
      { name: "get_job", description: "Check a job", inputSchema: { type: "object" } },
      { name: "provide_input", description: "Answer", inputSchema: { type: "object" } },
      { name: "cancel_job", description: "Cancel", inputSchema: { type: "object" } },
    ],
    call(tool, args) {
      if (tool === name) {
        const input = args?.input ?? {};
        if (effect === "read") return runOnce(input);
        // A write's first tick runs as a job, the way a slow site action does.
        job = { id: `job_${randomUUID()}`, ticks: 0, output: runOnce(input) };
        return { job_id: job.id, status: "running", next: { tool: "get_job", arguments: { job_id: job.id } } };
      }
      if (job === undefined) throw new Error("unknown job");
      if (tool === "get_job") {
        job.ticks += 1;
        return job.ticks === 1
          ? { job_id: job.id, status: "running" }
          : {
              job_id: job.id,
              status: "completed",
              output: job.output,
            };
      }
      if (tool === "cancel_job") return { job_id: job.id, status: "cancelled" };
      throw new Error(`unknown tool ${tool}`);
    },
  };
}

if (mode !== "serve" && mode !== "mint") {
  process.stderr.write("usage: fake-pomerado-mcp mint --root DIR | serve --artifact DIR\n");
  process.exit(2);
}
if (mode === "serve" && artifact === undefined) {
  process.stderr.write("--artifact is required\n");
  process.exit(2);
}
const tools = mode === "serve" ? await integrationTools(artifact) : mintTools();

const server = new Server({ name: "fake-pomerado", version: "0.0.0" }, { capabilities: { tools: {} } });
server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: tools.list }));
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  try {
    const output = await tools.call(request.params.name, request.params.arguments ?? {});
    return text(output);
  } catch (error) {
    return { isError: true, content: [{ type: "text", text: error.message }] };
  }
});
await server.connect(new StdioServerTransport());
