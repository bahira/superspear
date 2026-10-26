// The bot's actual work: mint an integration, then run it.
//
// Pomerado exposes both as jobs that may stop to ask a question. The flow here follows such
// a job to its end, relays every question to the agent that called the bot, and never
// resubmits anything: a job id is observed, not replayed.

import { PomeradoChild } from "./child.mjs";
import { archiveIntegration, readIntegration } from "./integrations.mjs";
import { mintCommand, runCommand } from "./config.mjs";
import { BotError, checkIntegrationName, checkUrl, siteSlug } from "./values.mjs";

const JOB_STATUSES = new Set(["running", "input_required", "completed", "failed", "cancelled"]);
const CHILD_WAIT_SECONDS = 25;

/** A Pomerado job view, as opposed to an operation's own output. */
export function jobView(value) {
  return value !== undefined &&
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    typeof value.job_id === "string" &&
    JOB_STATUSES.has(value.status)
    ? value
    : undefined;
}

function failureFromView(view) {
  const details = [];
  if (view.write_status !== undefined) details.push(`write_status: ${view.write_status}`);
  if (view.possible_commit === true) {
    details.push("a website action may have taken effect: read the site back before retrying");
  }
  if (view.retry === false) details.push("this call must not be retried");
  const message = `${view.error ?? "The operation failed."}${details.length === 0 ? "" : ` (${details.join("; ")})`}`;
  return new BotError(view.code ?? "operation_failed", message);
}

/**
 * Follow a Pomerado job to its end.
 *
 * `first` is whatever the child answered: an operation's output, or a job view. Questions
 * are published on the bot's own job through `ctx.ask`, so the calling agent answers them
 * with `provide_input` exactly as it would against Pomerado directly.
 */
export async function follow(ctx, child, first, { timeoutMs }) {
  let value = first;
  let view = jobView(value);
  if (view === undefined) return { output: value };
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    ctx.setChildJob(view.job_id);
    if (view.status === "completed") return { output: view.output };
    if (view.status === "failed") throw failureFromView(view);
    if (view.status === "cancelled") {
      const error = new Error("The job was cancelled.");
      error.code = "cancelled";
      throw error;
    }
    if (Date.now() > deadline) {
      throw new BotError(
        "timed_out",
        `The job ${view.job_id} did not finish in time. It may still be running on the website; check it before starting another one.`,
      );
    }
    if (view.status === "input_required") {
      if (view.pending_input === undefined) {
        throw new BotError("bad_child_job", "The job asked for input without describing the questions.");
      }
      ctx.progress(
        `waiting for the answer to ${Array.isArray(view.pending_input.questions) ? view.pending_input.questions.length : "some"} question(s)`,
      );
      const answers = await ctx.ask(view.pending_input);
      value = await child.call(
        "provide_input",
        { job_id: view.job_id, request_id: view.pending_input.id, answers },
        { timeoutMs: 120_000 },
      );
    } else {
      value = await child.call(
        "get_job",
        { job_id: view.job_id, wait_seconds: CHILD_WAIT_SECONDS },
        { timeoutMs: (CHILD_WAIT_SECONDS + 30) * 1000 },
      );
    }
    view = jobView(value) ?? { job_id: view.job_id, status: "completed", output: value };
  }
}

/** Mint one integration and leave the folder on disk. */
export async function mintIntegration(ctx, config, options, env) {
  const name = checkIntegrationName(options.name);
  const url = checkUrl(options.url);
  const task = options.task;
  const authority = options.authority;
  if (authority !== "read" && authority !== "write") {
    throw new BotError("bad_authority", 'authority must be "read" or "write".');
  }

  const command = mintCommand(config);
  ctx.progress(`starting the minter: ${command.command} ${command.args.join(" ")}`);
  const child = await PomeradoChild.connect({
    command: [command.command, ...command.args],
    env,
    label: "Pomerado minter",
  });
  ctx.attachChild(child);
  try {
    ctx.progress(`minting "${name}" from ${url} (${authority})`);
    const args = { name, url, intent: task, effect: authority };
    if (options.input !== undefined && Object.keys(options.input).length > 0) args.input = options.input;
    if (Array.isArray(options.signInOrigins) && options.signInOrigins.length > 0) {
      args.authenticationOrigins = options.signInOrigins;
    }
    const first = await child.call("mint", args, { timeoutMs: 120_000 });
    let output;
    try {
      ({ output } = await follow(ctx, child, first, { timeoutMs: config.mintTimeoutMs }));
    } catch (error) {
      // The most common mint failure by far is a missing key; name it instead of making the
      // agent guess from the minter's generic message.
      if (!config.modelKeyPresent) {
        throw new BotError(
          error.code ?? "mint_failed",
          `${error.message} OPENAI_API_KEY is not set in the bot's environment, and minting needs it. ` +
            "Run browser_doctor, or run the existing integration instead.",
          error,
        );
      }
      throw error;
    }
    if (output?.integration?.directory === undefined) {
      throw new BotError(
        "mint_incomplete",
        `The mint did not publish an integration${output?.summary ? `: ${output.summary}` : "."}`,
      );
    }
    ctx.progress(`minted ${name} into ${output.integration.directory}`);
    return { name, directory: output.integration.directory, output };
  } finally {
    const minting = ctx.detachChild();
    if (minting !== undefined) await minting.close().catch(() => {});
  }
}

/** Run one existing integration with the given input. */
export async function runIntegration(ctx, config, options, env) {
  const name = checkIntegrationName(options.name);
  const integration = await readIntegration(config.root, name);
  if (integration === undefined) {
    throw new BotError(
      "unknown_integration",
      `No integration named "${name}" under ${config.root}. Mint one with browser_mint, or use browser_do to mint and run in one step.`,
    );
  }
  const command = runCommand(config, integration.directory, integration.mcpJson);
  ctx.progress(`serving ${name} from ${integration.directory}`);
  const child = await PomeradoChild.connect({
    command: [command.command, ...command.args],
    env,
    label: `Integration ${name}`,
  });
  ctx.attachChild(child);
  try {
    const tools = await child.listTools();
    const tool =
      tools.find((candidate) => candidate.name === integration.deployment.name) ??
      tools.find((candidate) => !["get_job", "provide_input", "cancel_job"].includes(candidate.name));
    if (tool === undefined) {
      throw new BotError(
        "integration_broken",
        `The integration "${name}" exposes no operation tool. Re-mint it, or check its source under ${integration.directory}.`,
      );
    }
    ctx.setIntegration({
      integration: integration.name,
      url: integration.url,
      authority: integration.effect,
      directory: integration.directory,
    });
    ctx.progress(`calling ${tool.name}`);
    const first = await child.call(tool.name, { input: options.input ?? {} }, { timeoutMs: 180_000 });
    const { output } = await follow(ctx, child, first, { timeoutMs: config.runTimeoutMs });
    return output;
  } finally {
    const running = ctx.detachChild();
    if (running !== undefined) await running.close().catch(() => {});
  }
}

function sameSite(a, b) {
  if (typeof a !== "string") return false;
  try {
    return new URL(a).host === new URL(b).host;
  } catch {
    return false;
  }
}

function matches(integration, url, authority) {
  return integration !== undefined && sameSite(integration.url, url) && integration.effect === authority;
}

function suffixed(name, suffix) {
  const room = 60 - suffix.length - 1;
  return `${name.slice(0, Math.max(1, room))}_${suffix}`;
}

/**
 * Decide which folder a `browser_do` request should use.
 *
 * A named integration that does not match the request is never silently replaced: the
 * caller either gets a new folder name, or is told to pass `fresh: true`.
 */
async function resolveTarget(ctx, config, { url, authority, integration, fresh }) {
  const explicit = integration === undefined ? undefined : checkIntegrationName(integration);
  const slug = explicit ?? siteSlug(url);
  let existing = await readIntegration(config.root, slug);

  if (existing !== undefined && fresh === true) {
    ctx.progress(`archiving the previous "${slug}" before re-minting`);
    await archiveIntegration(config.root, slug);
    existing = undefined;
  }

  if (matches(existing, url, authority)) return { name: existing.name, reuse: true };

  if (existing !== undefined && explicit !== undefined) {
    throw new BotError(
      "integration_mismatch",
      `"${slug}" already exists for ${existing.url} with ${existing.effect} authority. ` +
        `Call browser_run to use it, pass "fresh": true to replace it, or pick another integration name. Nothing was changed.`,
    );
  }

  if (existing !== undefined) {
    // Same site, different authority: keep both rather than destroying the read-only one.
    const alternative = suffixed(slug, authority);
    const other = await readIntegration(config.root, alternative);
    if (matches(other, url, authority)) return { name: other.name, reuse: true };
    if (other !== undefined) {
      throw new BotError(
        "integration_mismatch",
        `"${alternative}" exists for ${other.url} with ${other.effect} authority. Pass "fresh": true to replace it, or choose an integration name.`,
      );
    }
    return { name: alternative, reuse: false };
  }

  return { name: slug, reuse: false };
}

/** The one call that does everything: reuse an integration, or mint one, then run it. */
export async function doTask(ctx, config, options, env) {
  const url = checkUrl(options.url);
  const authority = options.authority ?? "read";
  if (authority !== "read" && authority !== "write") {
    throw new BotError("bad_authority", 'authority must be "read" or "write".');
  }
  const target = await resolveTarget(ctx, config, {
    url,
    authority,
    integration: options.integration,
    fresh: options.fresh === true,
  });

  if (!target.reuse) {
    await mintIntegration(
      ctx,
      config,
      {
        name: target.name,
        url,
        task: options.task,
        authority,
        input: options.input,
        signInOrigins: options.signInOrigins,
      },
      env,
    );
  } else {
    ctx.progress(`reusing the existing integration "${target.name}"`);
  }

  const output = await runIntegration(
    ctx,
    config,
    { name: target.name, input: options.input },
    env,
  );
  const integration = await readIntegration(config.root, target.name);
  return {
    output,
    integration:
      integration === undefined
        ? { integration: target.name }
        : {
            integration: integration.name,
            url: integration.url,
            authority: integration.effect,
            task: integration.intent,
            directory: integration.directory,
          },
  };
}
