# Local draw.io MCP

Cursor's MCP host runs **Node 22**, where `npx drawio-mcp` crashes:

```
TypeError: Cannot set property navigator of #<Object> which has only a getter
```

This folder vendors `drawio-mcp@1.6.0` and patches that assignment.

After clone:

```bash
cd tools/drawio-mcp
npm install
```

`postinstall` re-applies the patch. `.cursor/mcp.json` starts it with:

```bash
node scripts/run-drawio-mcp.mjs
```
