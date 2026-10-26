// A readiness check, because every failure this bot has is an environment failure.
//
// Pomerado needs a recent Node, Chromium and an OpenAI key to mint; a run needs Node and
// Chromium only. A doctor that names the missing piece saves a twenty-minute mint that was
// never going to start.

import { spawn } from "node:child_process";
import { access, mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import { constants } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";

function runOnce(command, args, { timeoutMs = 120_000, env } = {}) {
  return new Promise((resolve) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"], env });
    let output = "";
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      finish({ ok: false, timedOut: true, output: output.slice(-800) });
    }, timeoutMs);
    child.stdout.on("data", (chunk) => {
      output += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      output += String(chunk);
    });
    child.on("error", (cause) => finish({ ok: false, error: cause.message, output }));
    child.on("close", (code) => finish({ ok: code === 0, code, output: output.slice(-800) }));
  });
}

function browserCacheDirectories() {
  const explicit = process.env.PLAYWRIGHT_BROWSERS_PATH;
  if (explicit !== undefined && explicit !== "" && explicit !== "0") return [explicit];
  return [
    join(homedir(), ".cache", "ms-playwright"),
    join(homedir(), "Library", "Caches", "ms-playwright"),
  ];
}

async function findChromium() {
  for (const directory of browserCacheDirectories()) {
    const entries = await readdir(directory).catch(() => undefined);
    if (entries === undefined) continue;
    const chromium = entries.filter((entry) => entry.startsWith("chromium-") || entry.startsWith("chromium_headless_shell-"));
    if (chromium.length > 0) return { installed: true, directory, builds: chromium };
  }
  return { installed: false, looked_in: browserCacheDirectories() };
}

function nodeVersionOk(version) {
  const [major, minor] = version.split(".").map((part) => Number.parseInt(part, 10));
  if (major === 24) return minor >= 21;
  return major > 24;
}

export async function doctor(config, env) {
  const checks = [];

  const nodeOk = nodeVersionOk(process.versions.node);
  checks.push({
    check: "node",
    ok: nodeOk,
    detail: nodeOk
      ? `Node ${process.versions.node}.`
      : `Node ${process.versions.node} is too old for Pomerado, which needs Node 24.21 or later in Node 24. The bot itself runs on Node 20+.`,
  });

  let rootOk = true;
  let rootDetail = `Integrations are saved under ${config.root}.`;
  try {
    await access(config.root, constants.W_OK);
  } catch {
    try {
      await rm(await mkdtemp(join(tmpdir(), "browser-bot-")), { recursive: true, force: true });
      const probe = join(config.root, ".write-probe");
      await writeFile(probe, "");
      await rm(probe, { force: true });
      rootDetail = `Created ${config.root}; it is writable.`;
    } catch (cause) {
      rootOk = false;
      rootDetail = `Cannot write to ${config.root}: ${cause.message}`;
    }
  }
  checks.push({ check: "integration_root", ok: rootOk, detail: rootDetail });

  checks.push({
    check: "model_key",
    ok: config.modelKeyPresent,
    detail: config.modelKeyPresent
      ? "OPENAI_API_KEY is set in the bot's environment; minting can call the model."
      : "OPENAI_API_KEY is not set. Listing and running existing integrations still works; minting a new one will fail.",
  });

  const help = await runOnce(config.pomerado[0], [...config.pomerado.slice(1), "--help"], {
    timeoutMs: 180_000,
    env,
  });
  checks.push({
    check: "pomerado",
    ok: help.ok,
    detail: help.ok
      ? `${config.pomeradoLabel} answers --help.`
      : `${config.pomeradoLabel} did not run (${help.error ?? (help.timedOut ? "timed out" : `exit ${help.code}`)}). ` +
        "Install it, or set BROWSER_BOT_POMERADO. " +
        (help.output ? `Output: ${help.output}` : ""),
  });

  const chromium = await findChromium();
  checks.push({
    check: "chromium",
    ok: chromium.installed,
    detail: chromium.installed
      ? `Chromium is installed in ${chromium.directory}.`
      : `No Chromium found in ${chromium.looked_in.join(", ")}. Run: npx -y -p pomerado playwright install chromium`,
  });

  return {
    ready: checks.every((entry) => entry.ok),
    ready_to_mint: checks.every((entry) => entry.ok),
    ready_to_run: checks.filter((entry) => entry.check !== "model_key").every((entry) => entry.ok),
    configuration: {
      integration_root: config.root,
      pomerado: config.pomeradoLabel,
      run_mode: config.runMode,
      default_wait_ms: config.waitMs,
      max_jobs: config.maxJobs,
      mint_timeout_ms: config.mintTimeoutMs,
      run_timeout_ms: config.runTimeoutMs,
    },
    checks,
  };
}
