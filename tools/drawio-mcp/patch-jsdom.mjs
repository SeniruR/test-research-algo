/**
 * drawio-mcp 1.6.0 assigns global.navigator, which is a getter in Node 22+.
 * Rewrite that bootstrap so the MCP server can start.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = dirname(fileURLToPath(import.meta.url));
const target = join(root, "node_modules", "drawio-mcp", "dist", "mxgraph", "jsdom.js");
const source = readFileSync(target, "utf8");
if (source.includes("Object.defineProperty(global, \"navigator\"")) {
  console.log("jsdom.js already patched");
  process.exit(0);
}

const patched = source.replace(
  `global.navigator = window.navigator;
global.location = window.location;`,
  `Object.defineProperty(global, "navigator", { value: window.navigator, writable: true, configurable: true });
Object.defineProperty(global, "location", { value: window.location, writable: true, configurable: true });`,
);

if (patched === source) {
  console.error("Could not patch", target);
  process.exit(1);
}
writeFileSync(target, patched);
console.log("patched", target);
