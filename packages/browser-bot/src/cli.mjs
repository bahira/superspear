// A command line for the same bot, so a person can try a task, or check an installation,
// without wiring an MCP client first. It speaks to the same tools the MCP server exposes.

import { parseArgs } from "node:util";
import { readFile } from "node:fs/promises";
import { loadConfig } from "./config.mjs";
import { JobStore } from "./jobs.mjs";
import { makeHandlers } from "./tools.mjs";
import { askInTerminal } from "./answers.mjs";

const USAGE = `browser-bot — do browser tasks through Pomerado integrations

  browser-bot serve                     Run the MCP server on stdio (the default)
  browser-bot doctor                    Check Node, Chromium, Pomerado and the model key
  browser-bot list                      List the integrations on this machine
  browser-bot describe <name>           Show an integration's input and output schema
  browser-bot do --url U --task T       Reuse or build an integration, then run it
  browser-bot mint --url U --task T     Build an integration without running it
  browser-bot run <name> [--input JSON] Run an existing integration
  browser-bot forget <name> --yes       Move an integration into .archive

Options
  --authority read|write   What the task may do (default read; write needs --approved)
  --approved               Confirm that the user agreed to a write
  --input JSON             Input for a mint's example or a run's values
  --integration NAME       Integration folder name
  --fresh                  Re-mint even when an integration already matches
  --sign-in-origin ORIGIN  Extra https origin the sign-in may use (repeatable)
  --root DIRECTORY         Where integrations live (default ~/browser-bot/integrations)
  --json                   Print only the final JSON

Environment
  BROWSER_BOT_ROOT, BROWSER_BOT_POMERADO, BROWSER_BOT_RUN_MODE,
  BROWSER_BOT_WAIT_MS, BROWSER_BOT_MAX_JOBS, OPENAI_API_KEY (minting only)
`;

function parse(argv) {
  return parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      help: { type: "boolean", short: "h" },
      url: { type: "string" },
      task: { type: "string" },
      authority: { type: "string" },
      approved: { type: "boolean" },
      input: { type: "string" },
      integration: { type: "string" },
      fresh: { type: "boolean" },
      "sign-in-origin": { type: "string", multiple: true },
      root: { type: "string" },
      yes: { type: "boolean" },
      json: { type: "boolean" },
      "wait-seconds": { type: "string" },
    },
  });
}

async function readInput(value) {
  if (value === undefined) return {};
  if (value.startsWith("@")) {
    const text = await readFile(value.slice(1), "utf8");
    return JSON.parse(text);
  }
  try {
    return JSON.parse(value);
  } catch (cause) {
    throw new Error(`--input must be JSON, or @path to a JSON file: ${cause.message}`);
  }
}

/**
 * Follow a job to its end in the terminal, asking its questions on stdin. This is the
 * command-line equivalent of an agent calling get_job and provide_input.
 */
async function followJob(jobs, first, { json }) {
  let view = first;
  let printed = 0;
  for (;;) {
    if (!json) {
      const steps = view.steps ?? [];
      if (steps.length < printed) printed = 0;
      for (const step of steps.slice(printed)) process.stderr.write(`· ${step}\n`);
      printed = steps.length;
    }
    if (view.status === "input_required") {
      const answers = await askInTerminal(view.pending_input);
      view = await jobs.provideInput(view.job_id, view.pending_input.id, answers);
      continue;
    }
    if (view.status === "running") {
      view = await jobs.get(view.job_id, 2_000);
      continue;
    }
    return view;
  }
}

export async function main(argv, env = process.env) {
  const { values, positionals } = parse(argv);
  if (values.help) {
    process.stdout.write(USAGE);
    return 0;
  }
  const command = positionals[0] ?? "serve";
  const config = loadConfig({ ...env, ...(values.root === undefined ? {} : { BROWSER_BOT_ROOT: values.root }) });

  if (command === "serve") {
    const { serveStdio } = await import("./server.mjs");
    await serveStdio({ config, env });
    return undefined;
  }

  const jobs = new JobStore({ maxJobs: config.maxJobs });
  const handlers = makeHandlers({ config, jobs, env });
  const print = (payload) => {
    process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
  };
  const authority = values.authority ?? "read";
  const wait = Number(values["wait-seconds"] ?? 0);

  switch (command) {
    case "doctor": {
      const report = await handlers.browser_doctor();
      if (values.json) print(report);
      else {
        process.stdout.write(`Integration root: ${report.configuration.integration_root}\n`);
        for (const check of report.checks) {
          process.stdout.write(`${check.ok ? "ok  " : "FAIL"} ${check.check}: ${check.detail}\n`);
        }
        process.stdout.write(
          `\n${report.ready ? "Ready to mint and run." : report.ready_to_run ? "Ready to run existing integrations; minting is blocked." : "Not ready."}\n`,
        );
      }
      return report.ready_to_run ? 0 : 1;
    }
    case "list": {
      print(await handlers.browser_list());
      return 0;
    }
    case "describe": {
      const name = positionals[1] ?? values.integration;
      print(await handlers.browser_describe({ integration: name }));
      return 0;
    }
    case "forget": {
      const name = positionals[1] ?? values.integration;
      print(await handlers.browser_forget({ integration: name, confirm: values.yes === true }));
      return 0;
    }
    case "run": {
      const name = positionals[1] ?? values.integration;
      const input = await readInput(values.input);
      const started = await handlers.browser_run({ integration: name, input, wait_seconds: wait });
      const final = await followJob(jobs, started, { json: values.json === true });
      print(values.json ? final.output ?? final : final);
      return final.status === "completed" ? 0 : 1;
    }
    case "mint":
    case "do": {
      const args = {
        url: values.url,
        task: values.task,
        authority,
        approved_by_user: values.approved === true,
        input: await readInput(values.input),
        integration: values.integration,
        sign_in_origins: values["sign-in-origin"],
        fresh: values.fresh === true,
        wait_seconds: wait,
      };
      const started =
        command === "mint" ? await handlers.browser_mint(args) : await handlers.browser_do(args);
      const final = await followJob(jobs, started, { json: values.json === true });
      print(values.json ? final.output ?? final : final);
      return final.status === "completed" ? 0 : 1;
    }
    default:
      process.stderr.write(`Unknown command "${command}".\n\n${USAGE}`);
      return 2;
  }
}
