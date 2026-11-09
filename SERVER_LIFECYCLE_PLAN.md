# Planned feature: managed local model-server lifecycle

**Status: design note only. Not implemented by the current `/debug` and offline-Wikipedia changes.**

The intended user-facing commands may eventually include natural phrases such as
`Start the server` and `Stop the server` (with equivalent aliases). The goal is
to reduce idle RAM and battery use on the target phone when Aria is only using
Crutch/Hassil or offline Wikipedia retrieval.

## Proposed behaviour

- `Start the server`: start or load the configured LFM model through an explicit
  server adapter, then verify that the configured model is actually ready.
- `Stop the server`: unload the model or stop only the server process Aria
  explicitly launched and owns. Do not terminate an unrelated LM Studio or
  llama-server process.
- `/status` should distinguish server process state, model loaded state, and
  API readiness. A reachable `/models` endpoint alone does not prove the
  intended model is loaded.
- Deterministic Crutch tools and offline Wikipedia must continue working while
  the model server is stopped.
- Ordinary model-dependent questions while stopped should explain that the
  model is not loaded and provide the explicit start command; they must not
  silently start a large model and consume battery.

## Implementation constraints for the future

1. Add a small provider interface rather than hard-coding LM Studio-specific
   behaviour into Crutch. Providers may include LM Studio and llama.cpp server.
2. Require an explicit configured model identifier/path and server executable
   or supported local management API. Do not guess which model to load.
3. Track whether Aria owns the process it starts. Never kill a process based
   only on a matching process name or port number.
4. Add readiness polling, startup/shutdown timeouts, clear errors, and a
   recoverable state if a load/unload operation fails.
5. Keep the GUI responsive by running lifecycle actions in a worker thread.
6. Add tests with mocked server adapters before testing on the target phone.
7. Consider an optional idle-unload timeout only after explicit start/stop
   commands work reliably; make the timeout configurable and disabled by
   default.

The current implementation does not start, stop, load, unload, or manage model
server processes. This document records the future direction only.
