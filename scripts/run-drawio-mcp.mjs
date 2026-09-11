#!/usr/bin/env node
/**
 * Start the locally installed drawio-mcp on Node 22+.
 * Cursor's MCP host uses its own Node (v22), where drawio-mcp's jsdom bootstrap
 * crashes on `global.navigator = ...`. Import the shim first, then the server.
 */
import "./node-navigator-shim.mjs";
await import("../tools/drawio-mcp/node_modules/drawio-mcp/dist/index.js");
