# ffmpegcv documentation

ffmpegcv is an OpenCV-compatible, pure-Python video reader/writer backed by the `ffmpeg`
CLI. `numpy` is the only required Python dependency.

## Start here

| Audience | Document |
| -------- | -------- |
| **AI coding agents** | [../AGENTS.md](../AGENTS.md) — rules, recipes, API map, repo layout |
| Claude Code users | [../CLAUDE.md](../CLAUDE.md) — short pointer to `AGENTS.md` |
| Humans (English) | [../README.md](../README.md) |
| Humans (中文) | [../README_CN.md](../README_CN.md) |
| Machine-readable index | [../llms.txt](../llms.txt) |

## Agent reference set (`docs/agents/`)

| Document | Contents |
| -------- | -------- |
| [quickstart.md](agents/quickstart.md) | 12 copy-paste recipes: read, write, copy, ROI, GPU, CUDA, camera, RTSP, noblock |
| [api-reference.md](agents/api-reference.md) | Exact signatures, options, returned objects, pixel-format table |
| [architecture.md](agents/architecture.md) | Module map, ffmpeg command templates, filter builders, process/thread model, CUDA path |
| [troubleshooting.md](agents/troubleshooting.md) | Error → cause → fix for import, GPU, shapes, writer, camera/stream issues |
| [testing.md](agents/testing.md) | Running and extending `tests/compat_suite.py`, JSON schema, determinism notes |

## Using ffmpegcv from an agent

Point any coding agent at [../AGENTS.md](../AGENTS.md); it contains the ground rules
(`ffmpeg` on `PATH`, always `release()`, read-only frames, even sizes) and runnable examples.
A typical prompt:

> Using this repository, add a function that reads `input.mp4` as RGB at 640×480 with aspect
> ratio preserved, and writes it back as H.264. Follow AGENTS.md and verify with the compat suite.

## Documentation conventions for contributors

- Agent docs are task-first: a runnable snippet, then the constraint that bites.
- Keep signatures in `api-reference.md` in sync with `ffmpegcv/__init__.py`.
- Keep command templates in `architecture.md` in sync with the reader/writer modules.
- Link source files with relative paths so links work offline in editors.
