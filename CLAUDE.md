# CLAUDE.md

See [AGENTS.md](./AGENTS.md) — it is the single source of truth for agents working in this
repository. It covers the hard rules (`ffmpeg`/`ffprobe` on `PATH`, always `release()`,
read-only frames, even resize sizes), task recipes, the public API map and the definition of
done.

Deeper references live in [docs/agents/](docs/agents/):

- [quickstart.md](docs/agents/quickstart.md) — copy-paste recipes
- [api-reference.md](docs/agents/api-reference.md) — exact signatures
- [architecture.md](docs/agents/architecture.md) — module map and ffmpeg commands
- [troubleshooting.md](docs/agents/troubleshooting.md) — error → cause → fix
- [testing.md](docs/agents/testing.md) — running `tests/compat_suite.py`

Quick verification:

```bash
PYTHONPATH=. python tests/compat_suite.py --out /tmp/ffmpegcv_compat.json
```
