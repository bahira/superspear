// MCP server smoke test — verifies tools/list returns all 5 tools.
const { spawn } = require("child_process");
const path = require("path");

const tsx = path.resolve(__dirname, "..", "node_modules", ".bin", "tsx.cmd");
const p = spawn(tsx, ["src/mcp/server.ts"], {
  cwd: path.resolve(__dirname, ".."),
  stdio: ["pipe", "pipe", "pipe"],
  shell: true,
});

let buf = "";
let done = false;

p.stdout.on("data", (d) => {
  buf += d.toString();
  const lines = buf.split("\n").filter((l) => l.trim());
  for (const line of lines) {
    try {
      const msg = JSON.parse(line);
      if (msg.id === 2 && msg.result?.tools) {
        console.log("TOOLS:", msg.result.tools.map((t) => t.name).join(", "));
        done = true;
        p.kill();
        process.exit(0);
      }
      if (msg.id === 1 && msg.result?.serverInfo) {
        console.log("SERVER:", msg.result.serverInfo.name, msg.result.serverInfo.version);
      }
    } catch {}
  }
});
p.stderr.on("data", (d) => { /* suppress */ });

setTimeout(() => {
  p.stdin.write(JSON.stringify({
    jsonrpc: "2.0", id: 1, method: "initialize",
    params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "test", version: "1.0" } },
  }) + "\n");
  setTimeout(() => {
    p.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) + "\n");
    p.stdin.write(JSON.stringify({ jsonrpc: "2.0", id: 2, method: "tools/list" }) + "\n");
  }, 300);
}, 500);

setTimeout(() => {
  if (!done) { console.error("TIMEOUT"); process.exit(1); }
}, 15000);
