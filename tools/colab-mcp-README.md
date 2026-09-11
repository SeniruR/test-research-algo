# Colab MCP (local)

Official [`googlecolab/colab-mcp`](https://github.com/googlecolab/colab-mcp) stays at **0 tools** in Cursor: notebook tools are only registered after a browser session via `notifications/tools/list_changed`, which Cursor does not apply.

This directory is a clone of the [SebastianGilPinzon fork](https://github.com/SebastianGilPinzon/colab-mcp) that pre-registers the connect + notebook tools at startup.

```bash
git clone --depth 1 https://github.com/SebastianGilPinzon/colab-mcp.git tools/colab-mcp
```

`.cursor/mcp.json` runs it with `uv run --directory ... colab-mcp`.
