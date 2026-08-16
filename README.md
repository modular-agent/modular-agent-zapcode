# ZapCode Agents for Modular Agent

Execute sandboxed TypeScript-subset scripts in Modular Agent using [TheUncharted/zapcode](https://github.com/TheUncharted/zapcode), a Rust-native TypeScript interpreter. Scripts can call registered LLM tools, define new tools, and even declare whole custom nodes — all without filesystem, network, or environment access.

## Features

- **ZC Expr** — Run TypeScript-like expressions and scripts to transform, filter, and reshape data in agent workflows
- **ZC Runner** — Execute LLM-generated code with access to registered tools (code-mode)
- **ZC Tool** — Define an LLM tool whose implementation is a script
- **ZC Script** — A generic node whose ports, configs, and behavior are declared by a script

## Installation

Two changes to add this package to [`modular-agent-desktop`](https://github.com/modular-agent/modular-agent-desktop):

1. **`modular-agent-desktop/src-tauri/Cargo.toml`** — add dependency:

   ```toml
   modular-agent-zapcode = { path = "../../modular-agent-zapcode" }
   ```

2. **`modular-agent-desktop/src-tauri/src/lib.rs`** — add import:

   ```rust
   #[allow(unused_imports)]
   use modular_agent_zapcode;
   ```

## ZC Expr

Evaluates user-provided expressions and scripts through the ZapCode interpreter. Scripts receive input as the variable `value` and the value of the last expression becomes the output. Scripts are compiled fresh on each invocation; an empty script is a silent no-op.

Scripts can call any registered tool with `await callTool(name, args)`; a failed tool call aborts the script with an error. `console.log` output is written to the application log, tagged with the agent id (output after the first `callTool` is not captured).

### Configuration

| Config | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| expr | text | "" | TypeScript-like expression or script to evaluate. Empty script produces no output (silent no-op) |
| skip_unit | boolean | false | When true, suppress output if the script results in `undefined` or `null` |
| time_limit_ms | integer | 5000 | Time limit for each stretch of script execution between `callTool` calls |
| memory_limit_mb | integer | 32 | Script memory limit in megabytes |

### Ports

- **Input**: `value` — Value passed to the script as the `value` variable
- **Output**: `value` — Result of the last expression in the script

### Usage Example

Double the input value:

```ts
value * 2
```

If input is `21`, output is `42`.

Filter and transform an array:

```ts
value.filter(x => x.length > 3).map(x => x.toUpperCase())
```

If input is `["hi", "hello", "hey", "world"]`, output is `["HELLO", "WORLD"]`.

Filter with `skip_unit` — set `skip_unit` to `true` and use `null` (or fall through to `undefined`) to suppress output:

```ts
value > 0 ? value : null
```

If input is `5`, output is `5`. If input is `-3`, no output is emitted.

Call a registered tool:

```ts
const r = await callTool("web-search", { query: value });
r
```

## ZC Runner

Executes LLM-generated TypeScript code with access to registered tools — the code-mode counterpart of giving an LLM individual tools. Send generated code, as a string or a message whose text is the code, to the `script` port. The value of the last expression is emitted on `value`.

Tools selected by the `tools` patterns are callable from the script. Each selected tool whose name is a valid identifier is available as a global async function taking a single arguments object:

```ts
const r = await webSearch({ query: "rust" });
r.results.length
```

Every selected tool — including names that are not valid identifiers, such as `web-search` — can also be invoked as `await callTool("web-search", { query: "rust" })`. Both paths go through the same allowlist: tools not selected by the patterns cannot be called either way.

### Configuration

| Config | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| tools | text | "" | Newline-separated regex patterns selecting callable tools (same semantics as the LLM chat agents' `tools` config). Empty means the script can call no tools |
| strip_fences | boolean | true | Unwrap a single Markdown code fence around the input |
| time_limit_ms | integer | 5000 | Time limit for each stretch of script execution between tool calls |
| memory_limit_mb | integer | 32 | Script memory limit in megabytes |

### Ports

- **Input**: `script` — Code to execute, as a string or a message (its text is used). With `strip_fences` enabled, a reply consisting of a single Markdown code fence (e.g. a ts-tagged fence) is unwrapped automatically. Input that still contains prose or multiple fences is executed as-is
- **Output**: `value` — Value of the last expression in the script
- **Output**: `console` — Console output captured from the script, emitted before `value` and omitted when empty. Capture stops at the first tool call: anything logged after it is lost

### Self-Correction Flow

Compile and runtime errors fail the run and flow out of the node's `err` port. Wiring that port back into the chat agent turns the ZC Runner into a self-correcting loop:

```
Chat (LLM) ──message──▶ ZC Runner ──value──▶ downstream / back to chat
    ▲                       │
    └──────── err ──────────┘
```

1. The chat agent is prompted to answer with a single ts-tagged code fence.
2. The ZC Runner strips the fence and executes the code, dispatching tool calls.
3. On failure, the error text (e.g. `ZapCode compile error: …`) flows from `err` back into the chat agent; the LLM sees its own error, fixes the code, and retries.
4. On success, `value` carries the result onward.

## ZC Tool

Defines an LLM tool implemented as a script. While the agent is running, the tool is registered under `name` so LLM agents whose `tools` patterns match it can call it.

On each call the tool's arguments are bound as script variables: every top-level argument becomes a variable of the same name, and the whole argument object is also available as `args` (the only way to reach arguments whose names are not valid identifiers). The value of the last expression is the tool result. Script errors are returned to the calling LLM as an error tool result, so the model sees the message and can retry.

Scripts can invoke other registered tools with `await callTool(name, args)`, composing tools out of tools. There is no recursion guard, and the time limit does not stop recursion — each nested call gets its own time budget — so a tool that calls itself, directly or indirectly, hangs until the flow is cancelled.

`console.log` output is written to the application log, tagged with the tool name (output after the first `callTool` is not captured).

### Configuration

| Config | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| name | string | "" | Tool name; must match `^[a-zA-Z0-9_-]{1,64}$`. Empty uses the agent definition name |
| description | text | "" | What the tool does and when to use it. Sent to the LLM — a detailed description materially improves tool selection |
| parameters | object | {} | JSON Schema describing the tool's arguments |
| script | text | "" | Script executed on each tool call |
| time_limit_ms | integer | 5000 | Time limit for each stretch of script execution between `callTool` calls |
| memory_limit_mb | integer | 32 | Script memory limit in megabytes |

### Ports

- **Input**: (none)
- **Output**: (none — the tool runs inline when an LLM calls it)

### Usage Example

With `parameters` declaring properties `a` and `b`, and the script:

```ts
a + b
```

a call with `{"a": 1, "b": 2}` returns `3`.

## ZC Script

A generic node whose ports, configs, and behavior are defined by a script. The script declares the node's shape in a top-level `AGENT` object and implements its behavior in an `onInput(port, value)` function (the name `process` is reserved by the sandbox):

```ts
const AGENT = {
  inputs: ["value", "reset"],          // default: ["value"]
  outputs: ["avg"],                    // default: ["value"]
  configs: {
    window: { type: "integer", value: 5, title: "Window" },
  },
  api: 1,
};
function onInput(port, value) {
  if (port === "reset") { setState("samples", []); return; }
  const n = getConfig("window");
  const samples = (getState("samples") ?? []).concat([value]).slice(-n);
  setState("samples", samples);
  emit("avg", samples.reduce((a, b) => a + b) / samples.length);
}
```

When the script changes, `AGENT` is re-evaluated (with no host functions and a short time budget, so top-level code must be free of side effects) and the node's ports and config fields update immediately. A broken script does not kill the node: it keeps its last valid ports and configs, and the error is reported. `name`, `title`, `category`, and `description` keys in `AGENT` are accepted and ignored. Declared config fields appear in the inspector sorted by name; their saved values survive patch save/reload.

Inside `onInput` (plain or `async`), these host functions are available:

| Function | Behavior |
| -------- | -------- |
| `emit(port, value)` | Send a value to a declared output port. Emits are collected and delivered after the script finishes; a failed run delivers nothing |
| `getConfig(name)` | Read a config value (`null` when unset) |
| `getState(key)` / `setState(key, value)` | Per-node state that persists across runs. In memory only — not saved with the patch |
| `log(message)` | Write a message to the application log |
| `callTool(name, args)` | Call a registered LLM tool and return its result |

### Configuration

| Config | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| script | text | "" | Script declaring the `AGENT` object and the `onInput` function. Script-declared configs appear as additional fields. An empty script leaves the node inert with the default ports |
| time_limit_ms | integer | 5000 | Time limit for each stretch of script execution between host-function calls; it does not bound the total duration of an `onInput` run |
| memory_limit_mb | integer | 32 | Memory budget for one `onInput` run in megabytes |

### Ports

- **Input**: `value` — Default input port; replaced by the `inputs` declared in `AGENT`
- **Output**: `value` — Default output port; replaced by the `outputs` declared in `AGENT`

### Usage Example

With the moving-average script above, sending `1`, `2`, `3` to `value` emits `1`, `1.5`, `2` on `avg`; sending anything to `reset` clears the window.

## Type Mapping

All four agents share the same value bridge.

**Input (AgentValue → ZapCode):**

| AgentValue | ZapCode Type | Notes |
| ---------- | ------------ | ----- |
| Unit | null | Scripts never receive `undefined` as an input |
| Boolean | boolean | |
| Integer | int | |
| Number | float | |
| String | string | |
| Array | array | Elements converted recursively |
| Object | object | Keys sorted alphabetically so scripts observe a stable property order |
| Tensor | array | f32 values widened to float |
| Message | object | Fields via `value.role`, `value.content`; block content becomes an array of block objects |
| Error | string | Formatted error string |
| Image | null | Image data is not accessible from scripts |

**Output (ZapCode → AgentValue):**

| ZapCode Type | AgentValue | Notes |
| ------------ | ---------- | ----- |
| undefined, null | Unit | Suppressed when `skip_unit` is true (ZC Expr) |
| boolean | Boolean | |
| int | Integer | |
| float | Number | |
| string | String | |
| array | Array | Elements converted recursively |
| object | Object | Property order is not preserved |
| function, generator, method | (error) | No data representation — the run fails with a hint that the last expression is the output ("did you forget to call the function?") |

Note the asymmetry: `Unit` maps **in** as `null` only, but both `undefined` and `null` map **out** to `Unit`. A `Unit` that round-trips through a script comes back as `Unit` either way, but downstream agents cannot tell whether a script produced `undefined` or `null`.

## Limitations

ZapCode is a sandboxed interpreter for a subset of TypeScript — not Node, not a browser:

- Subset language: see the [zapcode repository](https://github.com/TheUncharted/zapcode) for supported features
- The spread operator (`...`) is not implemented, and it fails silently: `[...xs, y]` produces the nested array `[xs, y]` instead of spreading (object spread and spread call arguments also misbehave). Use `xs.concat([y])` instead
- No `import`, `require`, or `eval` — no module system and no dynamic code loading
- No filesystem, network, or environment access; the only doors out of the sandbox are the host functions each agent declares (`callTool`, and for the ZC Script agent `emit` / `getConfig` / `getState` / `setState` / `log`)
- Resource limits: execution stops with an error when the time limit (`time_limit_ms`, wall clock) or memory limit (`memory_limit_mb`) is exceeded. The time limit applies to each stretch of script execution between host calls, so it does not bound the total duration of a run that makes tool calls
- `console.log` capture stops at the first tool call; later output is lost
- Tool-from-tool recursion is not guarded (see ZC Tool)

## Error Handling

- **Compile errors** (`ZapCode compile error: …`) and **runtime errors** (`ZapCode runtime error: …`) fail the agent's `process()` and flow out of the node's `err` port.
- **ZC Tool** is the exception: its script errors become error tool results, which are returned to the calling LLM instead of failing the flow.
- A **failed tool call** inside a script (`callTool` or a direct tool function) aborts the script with the tool's error.

## Architecture

Each run confines the VM to a single `spawn_blocking` closure. When the script calls a host function the VM suspends; the call is bridged over a channel to the async side, executed there (this is where tool calls `.await`), and the result resumes the VM. Only `AgentValue`s cross the thread boundary. Scripts are compiled fresh on each invocation (no bytecode cache).

## Key Dependencies

- [zapcode](https://github.com/TheUncharted/zapcode) — Rust-native sandboxed TypeScript-subset interpreter

## License

Apache-2.0 OR MIT
