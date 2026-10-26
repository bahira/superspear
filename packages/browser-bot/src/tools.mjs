// The bot's tool surface.
//
// Two shapes of tool live here. `browser_do`, `browser_mint` and `browser_run` start a job
// and answer with its state, because the work takes minutes. The rest are quick reads over
// the integrations folder. All of them return JSON in `structuredContent` and the same JSON
// as text, so any client can read the answer.

import { archiveIntegration, describeIntegration, listIntegrations, readIntegration } from "./integrations.mjs";
import { doTask, mintIntegration, runIntegration } from "./flows.mjs";
import { doctor } from "./doctor.mjs";
import {
  BotError,
  InputError,
  checkIntegrationName,
  checkUrl,
  readBoolean,
  readObject,
  readSeconds,
  readString,
  siteSlug,
} from "./values.mjs";

const WAIT_PROPERTY = {
  type: "number",
  minimum: 0,
  maximum: 30,
  description: "Seconds to wait for the job before answering with its current state. Default 20.",
};

const INPUT_PROPERTY = {
  type: "object",
  description:
    "The operation's input, matching the integration's input schema. A mint uses it as the example the build starts from; a run uses it as the actual values.",
};

const APPROVAL_PROPERTY = {
  type: "boolean",
  description:
    "Required to be true for authority \"write\". Set it only after your user has explicitly approved changing something on the site in this conversation.",
};

const INTEGRATION_PROPERTY = {
  type: "string",
  pattern: "^[a-z][a-z0-9_]{0,59}$",
  description:
    "Integration folder name: lowercase letters, digits and underscores. Omit it to let the bot derive one from the site's host.",
};

const SIGN_IN_PROPERTY = {
  type: "array",
  items: { type: "string" },
  description:
    'Extra https origins the sign-in may use, as origins ("https://accounts.example.com"). Needed only when the site signs in on another domain.',
};

const READ_ONLY = { readOnlyHint: true, openWorldHint: false };
const CHANGES_SITE = {
  readOnlyHint: false,
  destructiveHint: true,
  idempotentHint: false,
  openWorldHint: true,
};

export const tools = [
  {
    name: "browser_do",
    title: "Do anything on a website",
    description:
      "The one call for a browser task. Give it a URL and the task in plain language; it reuses the " +
      "integration already built for that site, or mints one first, then runs it and returns the result. " +
      "A mint takes minutes and costs model tokens; later calls for the same site are deterministic and fast. " +
      'Authority "read" only looks at the site. Authority "write" changes something, and needs approved_by_user. ' +
      "If the site needs a login, the job stops and asks for the credentials through provide_input; the bot saves none of them.",
    inputSchema: {
      type: "object",
      properties: {
        url: { type: "string", description: "The page to start from, http or https." },
        task: {
          type: "string",
          description:
            'What to do, in plain language, names included: "Read the price of the first search result for X", "Book a table for two on Friday".',
        },
        authority: { type: "string", enum: ["read", "write"], default: "read" },
        approved_by_user: APPROVAL_PROPERTY,
        integration: INTEGRATION_PROPERTY,
        input: INPUT_PROPERTY,
        sign_in_origins: SIGN_IN_PROPERTY,
        fresh: {
          type: "boolean",
          description: "Re-mint even when an integration already matches. The old folder is archived, not deleted.",
        },
        wait_seconds: WAIT_PROPERTY,
      },
      required: ["url", "task"],
      additionalProperties: false,
    },
    annotations: { title: "Do anything on a website", ...CHANGES_SITE },
  },
  {
    name: "browser_mint",
    title: "Build an integration",
    description:
      "Build (mint) an integration for a site without running it. Use browser_do unless you deliberately " +
      "want to build now and run later, or want to inspect the input schema with browser_describe before calling it.",
    inputSchema: {
      type: "object",
      properties: {
        url: { type: "string" },
        task: { type: "string" },
        authority: { type: "string", enum: ["read", "write"] },
        approved_by_user: APPROVAL_PROPERTY,
        integration: INTEGRATION_PROPERTY,
        input: INPUT_PROPERTY,
        sign_in_origins: SIGN_IN_PROPERTY,
        wait_seconds: WAIT_PROPERTY,
      },
      required: ["url", "task", "authority"],
      additionalProperties: false,
    },
    annotations: { title: "Build an integration", ...CHANGES_SITE },
  },
  {
    name: "browser_run",
    title: "Run an integration",
    description:
      "Run an integration that already exists, with the input you pass. No model is called and no key is " +
      "needed. Use browser_describe first when you are unsure of the input shape. A write returns a job whose " +
      "outcome says whether the change was confirmed.",
    inputSchema: {
      type: "object",
      properties: {
        integration: { ...INTEGRATION_PROPERTY, description: "The integration to run." },
        input: INPUT_PROPERTY,
        wait_seconds: WAIT_PROPERTY,
      },
      required: ["integration"],
      additionalProperties: false,
    },
    annotations: { title: "Run an integration", ...CHANGES_SITE },
  },
  {
    name: "browser_list",
    title: "List integrations",
    description: "List the integrations on this machine with their site, authority and task.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { title: "List integrations", ...READ_ONLY },
  },
  {
    name: "browser_describe",
    title: "Describe an integration",
    description:
      "Show one integration's input and output schema, the questions a run may ask, and its sign-in origins, so " +
      "you can call it correctly the first time.",
    inputSchema: {
      type: "object",
      properties: { integration: { ...INTEGRATION_PROPERTY, description: "The integration to describe." } },
      required: ["integration"],
      additionalProperties: false,
    },
    annotations: { title: "Describe an integration", ...READ_ONLY },
  },
  {
    name: "browser_forget",
    title: "Archive an integration",
    description:
      "Move an integration out of the way into the root's .archive folder. Nothing is deleted, so a person can " +
      "restore it by hand. Use it when a site changed and the integration is stale.",
    inputSchema: {
      type: "object",
      properties: {
        integration: { ...INTEGRATION_PROPERTY, description: "The integration to archive." },
        confirm: { type: "boolean", description: "Must be true. Confirms the user wants it archived." },
      },
      required: ["integration", "confirm"],
      additionalProperties: false,
    },
    annotations: { title: "Archive an integration", readOnlyHint: false, destructiveHint: true, openWorldHint: false },
  },
  {
    name: "get_job",
    title: "Check a job",
    description:
      "Observe a job started by browser_do, browser_mint or browser_run. Waiting here never restarts work and " +
      "never repeats a website action.",
    inputSchema: {
      type: "object",
      properties: {
        job_id: { type: "string" },
        wait_seconds: WAIT_PROPERTY,
      },
      required: ["job_id"],
      additionalProperties: false,
    },
    annotations: { title: "Check a job", ...READ_ONLY },
  },
  {
    name: "provide_input",
    title: "Answer a job's questions",
    description:
      "Answer the questions a job is waiting on, keyed by question id. Choices use option ids; with allowOther, " +
      "{\"other\": text}; with allowNote, {\"option\": id, \"note\": text}; a multi_choice is an array of ids or " +
      "{\"options\": [ids]}. A secret answer is visible to this conversation: ask the user before sending a password " +
      "or a one-time code, and expect the site to ask again on the next call because nothing is saved.",
    inputSchema: {
      type: "object",
      properties: {
        job_id: { type: "string" },
        request_id: { type: "string", description: "The pending request id from get_job." },
        answers: { type: "object", additionalProperties: true, description: "Answers keyed by question id." },
      },
      required: ["job_id", "request_id", "answers"],
      additionalProperties: false,
    },
    annotations: { title: "Answer a job's questions", readOnlyHint: false, destructiveHint: true, openWorldHint: true },
  },
  {
    name: "cancel_job",
    title: "Cancel a job",
    description:
      "Stop a job and close its browser. An action already sent to a website may still have taken effect, so read " +
      "the site back before starting another write.",
    inputSchema: {
      type: "object",
      properties: { job_id: { type: "string" } },
      required: ["job_id"],
      additionalProperties: false,
    },
    annotations: { title: "Cancel a job", readOnlyHint: false, destructiveHint: true, openWorldHint: true },
  },
  {
    name: "browser_doctor",
    title: "Check the installation",
    description:
      "Check Node, Chromium, the Pomerado command, the integration folder and the model key, and say exactly what " +
      "is missing. Run it first when something does not work.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { title: "Check the installation", ...READ_ONLY },
  },
];

function asToolResult(payload) {
  const text = JSON.stringify(payload, null, 2) ?? "null";
  return {
    content: [{ type: "text", text }],
    ...(payload !== null && typeof payload === "object" && !Array.isArray(payload)
      ? { structuredContent: payload }
      : {}),
  };
}

function failure(error) {
  const code = error instanceof InputError ? "invalid_input" : (error?.code ?? "error");
  return {
    isError: true,
    content: [
      {
        type: "text",
        text: `${error?.message ?? String(error)}${code === "invalid_input" ? "" : ` [${code}]`}`,
      },
    ],
  };
}

/** authority "write" only runs with a recorded human approval on this very call. */
function requireWriteApproval(authority, approved) {
  if (authority !== "write") return;
  if (approved !== true) {
    throw new BotError(
      "write_needs_user_approval",
      "This task would change something on a website, so it needs your user's explicit approval. " +
        'Describe what will change, ask them, and only then call again with approved_by_user: true.',
    );
  }
}

function authorityOf(value) {
  const authority = value ?? "read";
  if (authority !== "read" && authority !== "write") {
    throw new InputError('"authority" must be "read" or "write".');
  }
  return authority;
}

function signInOrigins(value) {
  if (value === undefined || value === null) return undefined;
  if (!Array.isArray(value)) throw new InputError('"sign_in_origins" must be an array of https origins.');
  for (const origin of value) {
    if (typeof origin !== "string") throw new InputError('"sign_in_origins" entries must be strings.');
    let parsed;
    try {
      parsed = new URL(origin);
    } catch {
      throw new InputError(`"sign_in_origins" entry is not a URL: ${origin}`);
    }
    if (parsed.protocol !== "https:" || parsed.origin !== origin) {
      throw new InputError(
        `"sign_in_origins" entry must be an https origin with no path, as in https://accounts.example.com (got ${origin}).`,
      );
    }
  }
  return value;
}

async function startAndWait(context, { kind, label, executor, waitMs }) {
  const started = context.jobs.start({ kind, label, execute: executor });
  return context.jobs.get(started.job_id, waitMs);
}

export function makeHandlers({ config, jobs, env }) {
  const context = { config, jobs, env };

  return {
    async browser_do(args) {
      const url = checkUrl(args.url);
      const task = readString(args.task, "task");
      const authority = authorityOf(args.authority);
      requireWriteApproval(authority, readBoolean(args.approved_by_user, "approved_by_user", false));
      const input = readObject(args.input, "input");
      const integration = args.integration === undefined ? undefined : checkIntegrationName(args.integration);
      const origins = signInOrigins(args.sign_in_origins);
      const fresh = readBoolean(args.fresh, "fresh", false);
      const waitMs = readSeconds(args.wait_seconds, config.waitMs);
      const label = `${authority} ${new URL(url).host}${integration === undefined ? "" : ` ${integration}`}`;
      return startAndWait(context, {
        kind: "do",
        label,
        waitMs,
        executor: (ctx) =>
          doTask(ctx, config, { url, task, authority, input, integration, signInOrigins: origins, fresh }, env),
      });
    },

    async browser_mint(args) {
      const url = checkUrl(args.url);
      const task = readString(args.task, "task");
      const authority = authorityOf(args.authority);
      requireWriteApproval(authority, readBoolean(args.approved_by_user, "approved_by_user", false));
      const integration = args.integration === undefined ? undefined : checkIntegrationName(args.integration);
      const input = readObject(args.input, "input");
      const origins = signInOrigins(args.sign_in_origins);
      const waitMs = readSeconds(args.wait_seconds, config.waitMs);
      const name = integration ?? siteSlug(url);
      const label = `mint ${name}`;
      return startAndWait(context, {
        kind: "mint",
        label,
        waitMs,
        executor: async (ctx) => {
          await mintIntegration(
            ctx,
            config,
            { name, url, task, authority, input, signInOrigins: origins },
            env,
          );
          const integrationInfo = await readIntegration(config.root, name);
          return {
            output: { minted: name, directory: integrationInfo?.directory },
            integration:
              integrationInfo === undefined ? { integration: name } : describeIntegration(integrationInfo),
          };
        },
      });
    },

    async browser_run(args) {
      const integration = checkIntegrationName(args.integration);
      const input = readObject(args.input, "input");
      const waitMs = readSeconds(args.wait_seconds, config.waitMs);
      return startAndWait(context, {
        kind: "run",
        label: `run ${integration}`,
        waitMs,
        executor: async (ctx) => {
          const output = await runIntegration(ctx, config, { name: integration, input }, env);
          return { output, integration: { integration } };
        },
      });
    },

    async browser_list() {
      const integrations = await listIntegrations(config.root);
      return {
        root: config.root,
        count: integrations.length,
        integrations: integrations.map((integration) => ({
          integration: integration.name,
          url: integration.url,
          authority: integration.effect,
          task: integration.intent,
          description: integration.description,
        })),
      };
    },

    async browser_describe(args) {
      const integration = checkIntegrationName(args.integration);
      const found = await readIntegration(config.root, integration);
      if (found === undefined) {
        throw new BotError("unknown_integration", `No integration named "${integration}" under ${config.root}.`);
      }
      return describeIntegration(found);
    },

    async browser_forget(args) {
      const integration = checkIntegrationName(args.integration);
      if (readBoolean(args.confirm, "confirm", false) !== true) {
        throw new InputError('"confirm" must be true: this moves the integration out of the way.');
      }
      const found = await readIntegration(config.root, integration);
      if (found === undefined) {
        throw new BotError("unknown_integration", `No integration named "${integration}" under ${config.root}.`);
      }
      const moved = await archiveIntegration(config.root, integration);
      return { archived: integration, from: moved.from, to: moved.to };
    },

    async get_job(args) {
      const jobId = readString(args.job_id, "job_id");
      const waitMs = readSeconds(args.wait_seconds, config.waitMs);
      return jobs.get(jobId, waitMs);
    },

    async provide_input(args) {
      const jobId = readString(args.job_id, "job_id");
      const requestId = readString(args.request_id, "request_id");
      const answers = readObject(args.answers, "answers");
      return jobs.provideInput(jobId, requestId, answers);
    },

    async cancel_job(args) {
      const jobId = readString(args.job_id, "job_id");
      return jobs.cancel(jobId);
    },

    async browser_doctor() {
      return doctor(config, env);
    },
  };
}

export { asToolResult, failure };
