# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

ZapCode agents for Modular Agent: four agents (`ZC Expr`, `ZC Runner`, `ZC Tool`,
`ZC Script`, category `Script/ZapCode`) that run sandboxed TypeScript-subset scripts
via [zapcode](https://github.com/TheUncharted/zapcode).
Out-of-tree package, same shape as `modular-agent-monty`.

Module layout: `value.rs` (AgentValue ↔ zapcode Value bridge), `bridge.rs`
(suspend/resume loop, `ExternalHandler`), `expr.rs` (ZC Expr), `runner.rs` (ZC Runner),
`tool.rs` (ZC Tool), `script.rs` (ZC Script).

## Dependency Decisions

- **`zapcode-core` is a git dependency pinned to a tag** (`tag = "v1.5.3"`), not a
  crates.io version: crates.io lags the repository (1.5.1 at the time of writing; the
  suspend/resume API this crate is built on needs the newer code). A tag pin keeps the
  build reproducible; bump it deliberately, never track a branch.
- **`modular-agent-core` must stay a crates.io dependency** (`"0.26.0"`). A path
  dependency would link a second copy of core, and two copies mean two separate
  `inventory` registries — every agent silently disappears from the app. The consuming
  workspace's `[patch.crates-io]` redirects it to the in-tree core.
- `indexmap` is a direct dependency only because zapcode-core does not re-export it;
  any 2.x unifies with its pin.

## Channel-Bridge Design (bridge.rs)

zapcode has no `register_fn`: when a script calls an external function, the VM
suspends and hands back a snapshot; the host resumes it with the call's result. The
bridge in `run_zapcode`:

- The **VM lives entirely inside one `spawn_blocking` closure** — construction, run,
  every resume, and both value conversions. On each suspension the closure
  `blocking_send`s a `HostRequest` (name, args, oneshot for the reply) and blocks; the
  async side runs `ExternalHandler::call(...).await` (this is where `call_tool`
  awaits) and replies.
- **Only `AgentValue` crosses the channel.** `ZapcodeSnapshot` / `zapcode::Value`
  never move between threads, because their `Send`-ness is not a documented upstream
  guarantee. A probe test (`snapshot_and_value_are_send`) records that both are `Send`
  today; the design does not depend on it, but if that probe ever stops compiling, the
  confinement becomes load-bearing rather than defensive.
- Handler errors abort the run in v1 (no guest exception injection — see v2 notes).
- Upstream limitation: stdout captured after a resume is not exposed by the snapshot,
  so console capture stops at the first external call. Documented on every agent.

## Determinism: Object Keys Are Sorted

`im::HashMap` iteration order is nondeterministic across runs, so
`agent_value_to_zapcode` sorts object keys before building the script-side object.
Scripts therefore observe a stable property order (`Object.keys`, `JSON.stringify`,
iteration) regardless of how the AgentValue was built. The same reasoning makes
ZC Script sort script-declared configs by name: the declaration object arrives as an
`im::HashMap`, its written order is already lost, and sorting is the only deterministic
choice. Output direction loses order again (`Object` → `im::HashMap`).

## Resource Limits

All execution paths lift `max_allocations` to `usize::MAX`: it counts VM stack pushes
(an instruction-rate proxy, not memory), and the 100k default halts ordinary loops
within milliseconds. Time and memory are the effective budgets. The one exception is
ZC Script's describe run, which keeps strict defaults on purpose — evaluating
`AGENT` should be trivial, and the tight budget doubles as enforcement that top-level
code stays side-effect free.

`Vm::from_snapshot` rebuilds a fresh `ResourceTracker`, so the time limit restarts on
every resume: it bounds each stretch between host calls, not the whole run. That is why
tool recursion is not stopped by the time limit (documented, guard deferred to v2).

## v2 Notes (deliberately not done)

- **File-based scripted agents**: `*.agent.ts` files appearing in the palette as node
  types (the monty design doc's original form). The `AGENT` declaration format of
  ZC Script is identical on purpose, so this is additive.
- **Snapshot persistence**: postcard dump/load of a suspended VM to resume across
  process restarts.
- **Bytecode/AST cache**: scripts are currently compiled fresh on every invocation.
- **Streaming emit**: ZC Script `emit()` is collected and delivered after the run;
  live delivery would need output during suspension.
- **Guest exception injection**: turn handler errors into catchable script exceptions
  instead of aborting the run.
- **ZC Runner `value` data input port**: feed data alongside the generated code.
- **Tool recursion guard**: call-depth tracking for `callTool` chains.

## Build and Test

```bash
cargo check
cargo test
cargo fmt
cargo clippy
```

The tool registry is process-global, so tool.rs tests must use unique tool names.
