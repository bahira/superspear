// Reading the folder of integrations Pomerado minted.
//
// An integration is a directory of source code plus a few JSON files. The bot reads
// `deployment.json` (what the site and task are, and what authority was granted) and
// `pomerado.json` (the input and output schema, and the questions a run may ask). It never
// edits an integration's source: that code was reviewed by Pomerado's Guardian, and a bot
// that rewrites it would be running something nobody reviewed.

import { readdir, readFile, rename, stat, mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { BotError } from "./values.mjs";

async function readJson(path) {
  try {
    return JSON.parse(await readFile(path, "utf8"));
  } catch (cause) {
    if (cause?.code === "ENOENT") return undefined;
    throw new BotError("unreadable_integration", `Could not read ${path}: ${cause.message}`, cause);
  }
}

export function integrationDirectory(root, name) {
  return resolve(root, name);
}

/** One integration's public description, or undefined when the folder is not one. */
export async function readIntegration(root, name) {
  const directory = integrationDirectory(root, name);
  const info = await stat(directory).catch(() => undefined);
  if (info === undefined || !info.isDirectory()) return undefined;
  const deployment = await readJson(join(directory, "deployment.json"));
  if (deployment === undefined) return undefined;
  const pomerado = await readJson(join(directory, "pomerado.json"));
  const mcpFile = await readJson(join(directory, "mcp.json"));
  // mcp.json holds { mcpServers: { <name>: { command, args } } }; keep the entry itself.
  const servers = mcpFile?.mcpServers ?? {};
  const mcpJson =
    servers[deployment.name] ??
    (Object.keys(servers).length === 1 ? Object.values(servers)[0] : undefined);
  return {
    name: deployment.name ?? name,
    directory,
    deployment,
    pomerado,
    mcpJson,
    url: deployment.request?.url,
    effect: deployment.request?.effect,
    intent: deployment.request?.intent,
    description: deployment.description,
    inputSchema: pomerado?.inputSchema ?? pomerado?.input ?? undefined,
    outputSchema: pomerado?.outputSchema ?? pomerado?.output ?? undefined,
  };
}

/** Every integration under the root, newest first. */
export async function listIntegrations(root) {
  const entries = await readdir(root, { withFileTypes: true }).catch((cause) => {
    if (cause?.code === "ENOENT") return [];
    throw cause;
  });
  const found = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    // `.archive` and `.jobs` are the bot's own bookkeeping, not integrations.
    if (entry.name.startsWith(".")) continue;
    const integration = await readIntegration(root, entry.name);
    if (integration !== undefined) found.push(integration);
  }
  found.sort((a, b) => a.name.localeCompare(b.name));
  return found;
}

/** The integration that already covers this site, if any. */
export function findForSite(integrations, url) {
  const site = new URL(url);
  return integrations.find((integration) => {
    if (typeof integration.url !== "string") return false;
    try {
      const other = new URL(integration.url);
      return other.host === site.host;
    } catch {
      return false;
    }
  });
}

/**
 * Move an integration out of the way instead of deleting it.
 *
 * Minting refuses to write into an existing folder, and re-minting is sometimes the right
 * answer when a site changed. Archiving keeps the old source readable, and a person can put
 * it back by hand.
 */
export async function archiveIntegration(root, name, reason = "archived") {
  const from = integrationDirectory(root, name);
  const archiveRoot = join(root, ".archive");
  await mkdir(archiveRoot, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const to = join(archiveRoot, `${name}-${stamp}`);
  await rename(from, to);
  return { from, to, reason };
}

/** A short, agent-facing summary: no source code, no saved secrets. */
export function describeIntegration(integration) {
  return {
    integration: integration.name,
    url: integration.url,
    authority: integration.effect,
    task: integration.intent,
    description: integration.description,
    directory: integration.directory,
    input_schema: integration.inputSchema,
    output_schema: integration.outputSchema,
    questions: integration.pomerado?.questions ?? [],
    sign_in_origins: integration.deployment?.request?.authenticationOrigins ?? [],
  };
}
