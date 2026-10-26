import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readdir, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const BIN = fileURLToPath(new URL("../bin/browser-bot.mjs", import.meta.url));
const FIXTURE = fileURLToPath(new URL("./fixtures/fake-pomerado-mcp.mjs", import.meta.url));

const root = await mkdtemp(join(tmpdir(), "browser-bot-test-"));
const botEnv = {
  PATH: process.env.PATH ?? "/usr/bin:/bin",
  HOME: process.env.HOME ?? tmpdir(),
  BROWSER_BOT_ROOT: root,
  BROWSER_BOT_POMERADO: `${process.execPath} ${FIXTURE}`,
};

async function startBot(extraEnv = {}) {
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [BIN],
    env: { ...botEnv, ...extraEnv },
    stderr: "pipe",
  });
  const client = new Client({ name: "browser-bot-test", version: "0.0.0" }, { capabilities: {} });
  await client.connect(transport);
  return {
    client,
    async call(name, args = {}) {
      const result = await client.callTool({ name, arguments: args }, undefined, { timeout: 60_000 });
      const text = (result.content ?? []).filter((part) => part.type === "text").map((part) => part.text).join("");
      const payload = result.structuredContent ?? (text.startsWith("{") || text.startsWith("[") ? JSON.parse(text) : text);
      return { isError: result.isError === true, payload, text };
    },
    async close() {
      await client.close();
    },
  };
}

/** Wait for a job to reach `status`, one get_job at a time, the way an agent would. */
async function until(bot, jobId, status, { timeoutMs = 30_000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  let view = await bot.call("get_job", { job_id: jobId, wait_seconds: 5 });
  while (view.payload.status !== status) {
    if (Date.now() > deadline) throw new Error(`job ${jobId} stayed ${view.payload.status}, wanted ${status}`);
    if (["failed", "cancelled"].includes(view.payload.status)) {
      throw new Error(`job ${jobId} ended ${view.payload.status}: ${view.text}`);
    }
    view = await bot.call("get_job", { job_id: jobId, wait_seconds: 5 });
  }
  return view.payload;
}

async function mintCount() {
  const text = await readFile(join(root, ".mint-count"), "utf8").catch(() => "0");
  return Number.parseInt(text, 10);
}

const bot = await startBot();
test.after(async () => {
  await bot.close().catch(() => {});
  await rm(root, { recursive: true, force: true });
});

test("lists the bot's tools with JSON schemas", async () => {
  const { tools } = await bot.client.listTools();
  const names = tools.map((tool) => tool.name).sort();
  assert.deepEqual(names, [
    "browser_describe",
    "browser_do",
    "browser_doctor",
    "browser_forget",
    "browser_list",
    "browser_mint",
    "browser_run",
    "cancel_job",
    "get_job",
    "provide_input",
  ]);
  const doTool = tools.find((tool) => tool.name === "browser_do");
  assert.equal(doTool.inputSchema.type, "object");
  assert.deepEqual(doTool.inputSchema.required, ["url", "task"]);
  assert.equal(doTool.annotations.destructiveHint, true);
});

test("rejects URLs Pomerado would refuse", async () => {
  const bad = await bot.call("browser_do", { url: "ftp://example.com/x", task: "read" });
  assert.equal(bad.isError, true);
  assert.match(bad.text, /http or https/);

  const withCredentials = await bot.call("browser_do", {
    url: "https://user:pass@example.com/",
    task: "read",
  });
  assert.equal(withCredentials.isError, true);
  assert.match(withCredentials.text, /credentials/);
});

test("a read task mints, asks its question and returns output", async () => {
  const started = await bot.call("browser_do", {
    url: "https://example.com/products",
    task: "read the main heading",
    authority: "read",
    wait_seconds: 0,
  });
  assert.equal(started.isError, false);
  const asking = await until(bot, started.payload.job_id, "input_required");
  assert.equal(asking.pending_input.questions[0].id, "account");
  assert.equal(asking.next.tool, "provide_input");

  const answered = await bot.call("provide_input", {
    job_id: asking.job_id,
    request_id: asking.pending_input.id,
    answers: { account: "yes" },
  });
  assert.equal(answered.isError, false);

  const done = await until(bot, started.payload.job_id, "completed");
  assert.equal(done.output.heading, "heading from example.com");
  assert.equal(done.integration.integration, "example_com");
  assert.equal(done.integration.authority, "read");

  const deployment = JSON.parse(await readFile(join(root, "example_com", "deployment.json"), "utf8"));
  assert.equal(deployment.request.effect, "read");
  assert.equal(deployment.request.url, "https://example.com/products");
  assert.equal(await mintCount(), 1);

  const listed = await bot.call("browser_list");
  assert.equal(listed.payload.count, 1);
  assert.equal(listed.payload.integrations[0].integration, "example_com");

  const described = await bot.call("browser_describe", { integration: "example_com" });
  assert.equal(described.payload.input_schema.properties.query.type, "string");
});

test("a second call for the same site reuses the integration", async () => {
  const started = await bot.call("browser_do", {
    url: "https://example.com/other",
    task: "read the main heading again",
    wait_seconds: 0,
  });
  const done = await until(bot, started.payload.job_id, "completed");
  assert.equal(done.output.heading, "heading from example.com");
  assert.equal(await mintCount(), 1);
  assert.ok(done.steps.some((step) => step.includes("reusing the existing integration")));
});

test("browser_run runs an existing integration without a mint", async () => {
  const result = await bot.call("browser_run", {
    integration: "example_com",
    input: { query: "shoes" },
  });
  assert.equal(result.isError, false);
  assert.deepEqual(result.payload.output.input, { query: "shoes" });
  assert.equal(await mintCount(), 1);
});

test("a write needs the user's approval before anything is minted", async () => {
  const refused = await bot.call("browser_do", {
    url: "https://shop.example.com/cart",
    task: "add the first product to the cart",
    authority: "write",
  });
  assert.equal(refused.isError, true);
  assert.match(refused.text, /write_needs_user_approval|explicit approval/);
  assert.equal(await mintCount(), 1);
  await assert.rejects(stat(join(root, "shop_example_com")));
});

test("an approved write mints, asks and runs as a job", async () => {
  const started = await bot.call("browser_do", {
    url: "https://shop.example.com/cart",
    task: "add the first product to the cart",
    authority: "write",
    approved_by_user: true,
    wait_seconds: 0,
  });
  const asking = await until(bot, started.payload.job_id, "input_required");
  await bot.call("provide_input", {
    job_id: asking.job_id,
    request_id: asking.pending_input.id,
    answers: { account: "yes" },
  });
  const done = await until(bot, started.payload.job_id, "completed");
  assert.equal(done.output.effect, "write");
  assert.equal(done.integration.authority, "write");
  assert.equal(await mintCount(), 2);

  const again = await bot.call("browser_run", { integration: "shop_example_com", input: { query: "x" } });
  assert.equal(again.isError, false);
  assert.equal(again.payload.status, "completed");
  assert.equal(again.payload.output.write_status, "confirmed");
  assert.equal(await mintCount(), 2);
});

test("a named integration with another authority is not replaced silently", async () => {
  const mismatch = await bot.call("browser_do", {
    url: "https://shop.example.com/cart",
    task: "read the cart",
    authority: "read",
    integration: "shop_example_com",
  });
  assert.equal(mismatch.isError, true);
  assert.match(mismatch.text, /integration_mismatch/);
  assert.equal(await mintCount(), 2);
});

test("fresh re-mints and archives the previous folder", async () => {
  const installed = await readdir(root);
  assert.ok(installed.includes("example_com"));
  const started = await bot.call("browser_do", {
    url: "https://example.com/products",
    task: "read the main heading, again",
    authority: "read",
    integration: "example_com",
    fresh: true,
    wait_seconds: 0,
  });
  const asking = await until(bot, started.payload.job_id, "input_required");
  await bot.call("provide_input", {
    job_id: asking.job_id,
    request_id: asking.pending_input.id,
    answers: { account: "yes" },
  });
  const done = await until(bot, started.payload.job_id, "completed");
  assert.equal(done.output.heading, "heading from example.com");
  assert.equal(await mintCount(), 3);
  const archive = await readdir(join(root, ".archive"));
  assert.equal(archive.length, 1);
  assert.match(archive[0], /^example_com-/);
});

test("a job's answers must match the pending request", async () => {
  const started = await bot.call("browser_do", {
    url: "https://news.example.org/",
    task: "read the front page headline",
    wait_seconds: 0,
  });
  const asking = await until(bot, started.payload.job_id, "input_required");
  const stale = await bot.call("provide_input", {
    job_id: asking.job_id,
    request_id: "00000000-0000-4000-8000-000000000000",
    answers: { account: "yes" },
  });
  assert.equal(stale.isError, true);
  assert.match(stale.text, /stale_input/);

  await bot.call("provide_input", {
    job_id: asking.job_id,
    request_id: asking.pending_input.id,
    answers: { account: "yes" },
  });
  await until(bot, started.payload.job_id, "completed");
});

test("unknown jobs and unknown integrations fail clearly", async () => {
  const job = await bot.call("get_job", { job_id: "job_missing" });
  assert.equal(job.isError, true);
  assert.match(job.text, /unknown_job/);

  const run = await bot.call("browser_run", { integration: "never_minted" });
  assert.equal(run.isError, true);
  assert.match(run.text, /unknown_integration/);
});

test("cancel_job stops a job that waits on a question", async () => {
  const started = await bot.call("browser_do", {
    url: "https://slow.example.net/",
    task: "read the front page",
    wait_seconds: 0,
  });
  const asking = await until(bot, started.payload.job_id, "input_required");
  const cancelled = await bot.call("cancel_job", { job_id: asking.job_id });
  assert.equal(cancelled.isError, false);
  assert.equal(cancelled.payload.status, "cancelled");
  const after = await bot.call("get_job", { job_id: asking.job_id, wait_seconds: 0 });
  assert.equal(after.payload.status, "cancelled");
  // The cancelled mint left no folder behind.
  await assert.rejects(stat(join(root, "slow_example_net")));
});

test("forget archives instead of deleting", async () => {
  const archived = await bot.call("browser_forget", { integration: "news_example_org", confirm: true });
  assert.equal(archived.isError, false);
  assert.match(archived.payload.to, /\.archive/);
  await assert.rejects(stat(join(root, "news_example_org")));
  const listed = await bot.call("browser_list");
  assert.equal(listed.payload.integrations.some((entry) => entry.integration === "news_example_org"), false);
});

test("a run mode of launcher replays the integration's own mcp.json", async () => {
  const launcherBot = await startBot({ BROWSER_BOT_RUN_MODE: "launcher" });
  try {
    const result = await launcherBot.call("browser_run", { integration: "example_com", input: {} });
    assert.equal(result.isError, false, result.text);
    assert.equal(result.payload.output.heading, "heading from example.com");
  } finally {
    await launcherBot.close();
  }
});

test("doctor reports each prerequisite", async () => {
  const report = await bot.call("browser_doctor");
  assert.equal(report.isError, false);
  const checks = report.payload.checks.map((check) => check.check);
  assert.deepEqual(checks, ["node", "integration_root", "model_key", "pomerado", "chromium"]);
  const pomerado = report.payload.checks.find((check) => check.check === "pomerado");
  assert.equal(pomerado.ok, true, pomerado.detail);
  const rootCheck = report.payload.checks.find((check) => check.check === "integration_root");
  assert.equal(rootCheck.ok, true, rootCheck.detail);
  assert.equal(report.payload.configuration.integration_root, root);
});
