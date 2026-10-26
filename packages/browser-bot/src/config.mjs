// Configuration for the browser bot.
//
// Everything is optional except a Pomerado installation: the bot is a thin, opinionated
// shell around `pomerado-mcp`, so the folder where integrations live and the command that
// runs Pomerado are the only two things it has to know.

import { homedir } from "node:os";
import { isAbsolute, resolve } from "node:path";

/** Split a command line the way a shell would for the simple cases we accept. */
export function splitCommand(line) {
  const parts = [];
  let current = "";
  let quote = null;
  let has = false;
  for (const char of line.trim()) {
    if (quote !== null) {
      if (char === quote) quote = null;
      else current += char;
      continue;
    }
    if (char === '"' || char === "'") {
      quote = char;
      has = true;
      continue;
    }
    if (/\s/.test(char)) {
      if (has) parts.push(current);
      current = "";
      has = false;
      continue;
    }
    current += char;
    has = true;
  }
  if (quote !== null) throw new Error(`Unbalanced quote in command: ${line}`);
  if (has) parts.push(current);
  return parts;
}

function positiveInteger(value, fallback) {
  const parsed = Number.parseInt(value ?? "", 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

/**
 * Read the bot's configuration from the environment.
 *
 * BROWSER_BOT_ROOT               where integrations are saved (absolute)
 * BROWSER_BOT_POMERADO           command that runs `pomerado-mcp`
 * BROWSER_BOT_POMERADO_VERSION   package spec for the default npx command
 * BROWSER_BOT_RUN_MODE           "pomerado-mcp" (default) or "launcher"
 * BROWSER_BOT_WAIT_MS            default get_job wait, 0..30000
 * BROWSER_BOT_MAX_JOBS           how many jobs may run at once, 1..8
 * BROWSER_BOT_MINT_TIMEOUT_MS    how long a single mint may take (default 20 min)
 * BROWSER_BOT_RUN_TIMEOUT_MS     how long one run may take (default 10 min)
 */
export function loadConfig(env = process.env) {
  const rootValue = env.BROWSER_BOT_ROOT?.trim();
  const root = rootValue === undefined || rootValue === ""
    ? resolve(homedir(), "browser-bot", "integrations")
    : isAbsolute(rootValue)
      ? resolve(rootValue)
      : resolve(process.cwd(), rootValue);

  const pomeradoLine = env.BROWSER_BOT_POMERADO?.trim() || "";
  const pomerado = pomeradoLine === ""
    ? ["npx", "-y", "-p", env.BROWSER_BOT_POMERADO_VERSION?.trim() || "pomerado", "pomerado-mcp"]
    : splitCommand(pomeradoLine);
  if (pomerado.length === 0) throw new Error("BROWSER_BOT_POMERADO is empty.");

  const runMode = (env.BROWSER_BOT_RUN_MODE?.trim() || "pomerado-mcp").toLowerCase();
  if (runMode !== "pomerado-mcp" && runMode !== "launcher") {
    throw new Error('BROWSER_BOT_RUN_MODE must be "pomerado-mcp" or "launcher".');
  }

  const waitMs = Math.min(30_000, Math.max(0, positiveInteger(env.BROWSER_BOT_WAIT_MS, 20_000)));
  const maxJobs = Math.min(8, positiveInteger(env.BROWSER_BOT_MAX_JOBS, 2));
  const mintTimeoutMs = positiveInteger(env.BROWSER_BOT_MINT_TIMEOUT_MS, 20 * 60_000);
  const runTimeoutMs = positiveInteger(env.BROWSER_BOT_RUN_TIMEOUT_MS, 10 * 60_000);

  return {
    root,
    pomerado,
    pomeradoLabel: pomerado.join(" "),
    runMode,
    waitMs,
    maxJobs,
    mintTimeoutMs,
    runTimeoutMs,
    /** The model key Pomerado's minter needs. A run never needs one. */
    modelKeyPresent: Boolean(env.OPENAI_API_KEY?.trim()),
  };
}

/** `pomerado-mcp mint --root <root>`: the server that mints integrations. */
export function mintCommand(config) {
  return { command: config.pomerado[0], args: [...config.pomerado.slice(1), "mint", "--root", config.root] };
}

/**
 * How to serve an existing integration.
 *
 * By default the bot asks its own Pomerado command to serve the folder, so a cached npx
 * install can move without breaking anything. "launcher" instead replays the `mcp.json`
 * that Pomerado wrote next to the integration, which names the exact Node and runtime
 * that minted it.
 */
export function runCommand(config, directory, mcpJson) {
  if (config.runMode === "launcher" && mcpJson !== undefined) {
    return { command: mcpJson.command, args: mcpJson.args };
  }
  return {
    command: config.pomerado[0],
    args: [...config.pomerado.slice(1), "serve", "--artifact", directory],
  };
}

/** The environment children inherit. Only strings, never undefined. */
export function childEnv(env = process.env) {
  return Object.fromEntries(
    Object.entries(env).filter(([, value]) => typeof value === "string"),
  );
}
