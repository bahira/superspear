#!/usr/bin/env node
// Entry point. `browser-bot` with no argument is the MCP server; see --help for the rest.

import { main } from "../src/cli.mjs";

try {
  const code = await main(process.argv.slice(2));
  if (typeof code === "number") process.exitCode = code;
} catch (error) {
  process.stderr.write(`${error?.message ?? error}\n`);
  process.exitCode = 1;
}
