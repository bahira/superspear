import { test } from "node:test";
import assert from "node:assert/strict";
import { JobStore } from "../src/jobs.mjs";
import { splitCommand } from "../src/config.mjs";
const codeIs = (code) => (error) => error?.code === code && error?.name !== "AssertionError";

import {
  BotError,
  checkIntegrationName,
  checkUrl,
  InputError,
  parseToolResult,
  siteSlug,
} from "../src/values.mjs";

test("siteSlug is stable, lowercase and Pomerado-shaped", () => {
  assert.equal(siteSlug("https://www.example.com/products?x=1"), "example_com");
  assert.equal(siteSlug("https://news.ycombinator.com/"), "news_ycombinator_com");
  assert.equal(siteSlug("https://shop.example.co.uk:8443/a"), "shop_example_co_uk");
  assert.equal(siteSlug("https://123.example.com/"), "site_123_example_com");
  assert.equal(siteSlug("https://mint.example.com/"), "mint_example_com");
  for (const url of ["https://a.b/", "http://exemple.fr/", "https://very-long-subdomain.example-with-a-long-name.travel/"]) {
    assert.match(siteSlug(url), /^[a-z][a-z0-9_]{0,59}$/);
  }
  assert.throws(() => siteSlug("file:///etc/passwd"), InputError);
});

test("checkUrl refuses what Pomerado refuses", () => {
  assert.equal(checkUrl("https://example.com/a"), "https://example.com/a");
  assert.equal(checkUrl(" http://example.com "), "http://example.com/");
  assert.throws(() => checkUrl("https://example.com/a#x"), /fragment/);
  assert.throws(() => checkUrl("https://u:p@example.com/"), /credentials/);
  assert.throws(() => checkUrl("javascript:alert(1)"), /http or https|not a valid URL/);
  assert.throws(() => checkUrl(""), /required/);
  assert.throws(() => checkUrl(42), /required/);
});

test("checkIntegrationName keeps names to the folder shape", () => {
  assert.equal(checkIntegrationName("example_com"), "example_com");
  assert.throws(() => checkIntegrationName("Example"), InputError);
  assert.throws(() => checkIntegrationName("../etc"), InputError);
  assert.throws(() => checkIntegrationName("a".repeat(61)), InputError);
});

test("splitCommand understands quotes", () => {
  assert.deepEqual(splitCommand("npx -y -p pomerado pomerado-mcp"), ["npx", "-y", "-p", "pomerado", "pomerado-mcp"]);
  assert.deepEqual(splitCommand('node "/path with space/mcp.js" serve'), [
    "node",
    "/path with space/mcp.js",
    "serve",
  ]);
  assert.throws(() => splitCommand('node "unbalanced'), /Unbalanced/);
});

test("parseToolResult prefers structured content and unwraps result", () => {
  assert.deepEqual(parseToolResult({ structuredContent: { result: [1, 2] } }), [1, 2]);
  assert.deepEqual(parseToolResult({ structuredContent: { status: "completed", output: { a: 1 } } }), {
    status: "completed",
    output: { a: 1 },
  });
  assert.deepEqual(parseToolResult({ content: [{ type: "text", text: '{"a":1}' }] }), { a: 1 });
  assert.equal(parseToolResult({ content: [{ type: "text", text: "plain words" }] }), "plain words");
  assert.equal(parseToolResult({ content: [] }), undefined);
});

test("a job publishes its questions and takes the answer once", async () => {
  const jobs = new JobStore({ maxJobs: 1 });
  const started = jobs.start({
    kind: "test",
    label: "asking",
    execute: async (ctx) => {
      ctx.progress("about to ask");
      const answers = await ctx.ask({
        id: "req-1",
        questions: [{ id: "q", type: "text", prompt: "Value?" }],
      });
      return { output: { echo: answers.q } };
    },
  });
  // The executor asks before it yields, so the first view may already carry the question.
  assert.ok(["running", "input_required"].includes(started.status));

  const asking = await jobs.get(started.job_id, 5_000);
  assert.equal(asking.status, "input_required");
  assert.equal(asking.pending_input.id, "req-1");
  assert.deepEqual(asking.next.arguments, {
    job_id: started.job_id,
    request_id: "req-1",
  });

  assert.throws(() => jobs.provideInput(started.job_id, "other", { q: "x" }), codeIs("stale_input"));
  jobs.provideInput(started.job_id, "req-1", { q: "hello" });
  const done = await jobs.get(started.job_id, 5_000);
  assert.equal(done.status, "completed");
  assert.deepEqual(done.output, { echo: "hello" });
  assert.throws(() => jobs.provideInput(started.job_id, "req-1", { q: "again" }), codeIs("stale_input"));
});

test("a job that fails reports its code, and the store stays usable", async () => {
  const jobs = new JobStore({ maxJobs: 1 });
  const started = jobs.start({
    kind: "test",
    label: "failing",
    execute: async () => {
      throw new BotError("integration_mismatch", "no");
    },
  });
  const done = await jobs.get(started.job_id, 5_000);
  assert.equal(done.status, "failed");
  assert.equal(done.code, "integration_mismatch");
  assert.equal(done.error, "no");
});

test("cancelling a waiting job releases its asker", async () => {
  let observed;
  const jobs = new JobStore({ maxJobs: 1 });
  const started = jobs.start({
    kind: "test",
    label: "cancelled",
    execute: async (ctx) => {
      try {
        await ctx.ask({ id: "req-9", questions: [{ id: "q", type: "text", prompt: "?" }] });
      } catch (error) {
        observed = error.code;
        throw error;
      }
      return { output: "never" };
    },
  });
  await jobs.get(started.job_id, 5_000);
  const cancelled = await jobs.cancel(started.job_id);
  assert.equal(cancelled.status, "cancelled");
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(observed, "cancelled");
  const after = await jobs.get(started.job_id, 100);
  assert.equal(after.status, "cancelled");
  assert.equal((await jobs.cancel(started.job_id)).status, "cancelled");
});

test("the store refuses more concurrent jobs than its limit", async () => {
  const jobs = new JobStore({ maxJobs: 1 });
  const release = [];
  jobs.start({
    kind: "test",
    label: "one",
    execute: () => new Promise((resolve) => release.push(resolve)),
  });
  assert.throws(() => jobs.start({ kind: "test", label: "two", execute: async () => {} }), codeIs("busy"));
  release[0]({ output: "done" });
  await new Promise((resolve) => setTimeout(resolve, 20));
  const second = jobs.start({ kind: "test", label: "two", execute: async () => ({ output: 2 }) });
  assert.equal((await jobs.get(second.job_id, 5_000)).status, "completed");
});

test("unknown job ids are refused", async () => {
  const jobs = new JobStore({ maxJobs: 1 });
  await assert.rejects(async () => jobs.get("job_nope", 0), codeIs("unknown_job"));
});
