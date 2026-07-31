use std::collections::BTreeSet;

use modular_agent_core::{
    Agent, AgentContext, AgentData, AgentError, AgentOutput, AgentSpec, AgentValue, AgentValueMap,
    AsAgent, ModularAgent, async_trait, modular_agent,
    tool::{call_tool, list_tool_infos_patterns},
};
use zapcode_core::ResourceLimits;

use crate::bridge::{ExternalHandler, run_zapcode};

static CATEGORY: &str = "Script/ZapCode";

static PORT_SCRIPT: &str = "script";
static PORT_VALUE: &str = "value";
static PORT_CONSOLE: &str = "console";

static CONFIG_TOOLS: &str = "tools";
static CONFIG_STRIP_FENCES: &str = "strip_fences";
static CONFIG_TIME_LIMIT_MS: &str = "time_limit_ms";
static CONFIG_MEMORY_LIMIT_MB: &str = "memory_limit_mb";

static EXTERNAL_CALL_TOOL: &str = "callTool";

const DEFAULT_TIME_LIMIT_MS: i64 = 5000;
const DEFAULT_MEMORY_LIMIT_MB: i64 = 32;

/// Executes LLM-generated TypeScript code with access to registered tools.
///
/// Send generated code — a string, or a message whose text is the code — to the
/// `script` port. The code runs in a sandboxed TypeScript-subset interpreter
/// with no filesystem, network, or environment access, and the value of its
/// last expression becomes the `value` output. Compile and runtime errors fail
/// the run and flow out of the error port; wiring that port back into a chat
/// agent gives the LLM a chance to correct its own code.
///
/// Tools selected by the `tools` patterns are callable from the script. Each
/// selected tool whose name is a valid identifier is available as a global
/// async function taking a single arguments object, for example
/// `await webSearch({ query: "rust" })`. Every selected tool — including names
/// that are not valid identifiers, such as `web-search` — can also be invoked
/// as `await callTool("web-search", { query: "rust" })`. Tools not selected by
/// the patterns cannot be called either way.
///
/// # Ports
/// - Input `script`: Code to execute, as a string or a message (its text is
///   used). With `strip_fences` enabled, a reply consisting of a single
///   Markdown code fence (e.g. a ts-tagged fence) is unwrapped automatically
/// - Output `value`: Value of the last expression in the script
/// - Output `console`: Console output captured from the script, emitted before
///   `value` and omitted when empty. Capture stops at the first tool call:
///   anything logged after it is lost
///
/// # Configuration
/// - `tools`: Newline-separated regular expressions selecting which registered
///   tools the script may call (same semantics as the LLM chat agents'
///   `tools` config). Empty means the script can call no tools
/// - `strip_fences`: Unwrap a reply that is a single Markdown code fence
///   before executing it (default: true)
/// - `time_limit_ms`: Time limit in milliseconds for each stretch of script
///   execution between tool calls; the clock restarts after every tool call,
///   so it does not bound the total run (default: 5000)
/// - `memory_limit_mb`: Script memory limit in megabytes (default: 32)
///
/// # Example
/// Given a chat reply containing only a ts-tagged code fence around
/// `const r = await webSearch({ query: "rust" }); r.results.length`, with a
/// `tools` pattern matching `webSearch`, the fence is stripped, the search
/// tool is called, and the result count is emitted on `value`.
#[modular_agent(
    title = "ZC Runner",
    category = CATEGORY,
    inputs = [PORT_SCRIPT],
    outputs = [PORT_VALUE, PORT_CONSOLE],
    text_config(
        name = CONFIG_TOOLS,
        title = "Tools",
        description = "Newline-separated regex patterns selecting callable tools"
    ),
    boolean_config(
        name = CONFIG_STRIP_FENCES,
        title = "Strip Code Fences",
        default = true,
        description = "Unwrap a single Markdown code fence around the input",
        detail
    ),
    integer_config(
        name = CONFIG_TIME_LIMIT_MS,
        title = "Time Limit (ms)",
        default = 5000,
        detail
    ),
    integer_config(
        name = CONFIG_MEMORY_LIMIT_MB,
        title = "Memory Limit (MB)",
        default = 32,
        detail
    ),
)]
struct ZcRunnerAgent {
    data: AgentData,
}

#[async_trait]
impl AsAgent for ZcRunnerAgent {
    fn new(ma: ModularAgent, id: String, spec: AgentSpec) -> Result<Self, AgentError> {
        Ok(Self {
            data: AgentData::new(ma, id, spec),
        })
    }

    async fn process(
        &mut self,
        ctx: AgentContext,
        _port: String,
        value: AgentValue,
    ) -> Result<(), AgentError> {
        let config = self.configs()?;

        let text = match &value {
            AgentValue::String(s) => s.as_ref().clone(),
            AgentValue::Message(m) => m.text(),
            _ => {
                return Err(AgentError::InvalidValue(
                    "script input must be a string or a message".into(),
                ));
            }
        };
        let source = if config.get_bool_or(CONFIG_STRIP_FENCES, true) {
            strip_code_fence(&text).to_string()
        } else {
            text
        };
        if source.trim().is_empty() {
            return Ok(());
        }

        let (externals, allowed) = tool_externals(&config.get_string_or_default(CONFIG_TOOLS))?;
        let limits = ResourceLimits {
            time_limit_ms: config
                .get_integer_or(CONFIG_TIME_LIMIT_MS, DEFAULT_TIME_LIMIT_MS)
                .max(1) as u64,
            memory_limit_bytes: config
                .get_integer_or(CONFIG_MEMORY_LIMIT_MB, DEFAULT_MEMORY_LIMIT_MB)
                .max(1) as usize
                * 1024
                * 1024,
            // max_allocations counts VM stack pushes — an instruction-rate
            // proxy, not memory — and the 100k default halts ordinary loops
            // within milliseconds. The wall-clock time limit is the budget.
            max_allocations: usize::MAX,
            ..ResourceLimits::default()
        };

        let mut handler = ToolDispatcher {
            ctx: ctx.clone(),
            allowed,
        };
        let outcome = run_zapcode(source, vec![], externals, limits, &mut handler).await?;

        if !outcome.console.is_empty() {
            self.output(
                ctx.clone(),
                PORT_CONSOLE,
                AgentValue::string(outcome.console),
            )
            .await?;
        }
        self.output(ctx, PORT_VALUE, outcome.value).await
    }
}

/// Routes external calls from the script to the tool registry. Direct calls
/// (`webSearch({…})`) and `callTool("name", {…})` converge on the same
/// allowlist, so tools outside the `tools` patterns stay unreachable.
struct ToolDispatcher {
    ctx: AgentContext,
    allowed: BTreeSet<String>,
}

#[async_trait]
impl ExternalHandler for ToolDispatcher {
    async fn call(
        &mut self,
        name: String,
        args: Vec<AgentValue>,
    ) -> Result<AgentValue, AgentError> {
        let (tool_name, tool_args) = if name == EXTERNAL_CALL_TOOL {
            if args.len() > 2 {
                return Err(AgentError::InvalidValue(
                    "callTool takes a tool name and one arguments object".into(),
                ));
            }
            let mut args = args.into_iter();
            let tool_name = match args.next() {
                Some(AgentValue::String(s)) => s.as_ref().clone(),
                _ => {
                    return Err(AgentError::InvalidValue(
                        "callTool expects a tool name string as its first argument".into(),
                    ));
                }
            };
            (tool_name, args.next())
        } else {
            if args.len() > 1 {
                return Err(AgentError::InvalidValue(format!(
                    "`{name}` takes a single arguments object"
                )));
            }
            (name, args.into_iter().next())
        };

        if !self.allowed.contains(&tool_name) {
            return Err(AgentError::InvalidValue(format!(
                "tool `{tool_name}` does not match the tools config"
            )));
        }
        let tool_args = tool_args.unwrap_or_else(|| AgentValue::Object(AgentValueMap::new()));
        call_tool(self.ctx.clone(), &tool_name, tool_args).await
    }
}

/// Resolves the `tools` patterns into the externals to declare to the VM and
/// the set of tool names the dispatcher may call. `callTool` is always
/// declared; matched tools are additionally exposed under their own name when
/// it is a valid identifier.
fn tool_externals(patterns: &str) -> Result<(Vec<String>, BTreeSet<String>), AgentError> {
    let mut externals = vec![EXTERNAL_CALL_TOOL.to_string()];
    let mut allowed = BTreeSet::new();
    if patterns.is_empty() {
        return Ok((externals, allowed));
    }
    let infos = list_tool_infos_patterns(patterns).map_err(|e| {
        AgentError::InvalidConfig(format!("Invalid regex patterns in tools config: {e}"))
    })?;
    for info in infos {
        if !allowed.insert(info.name.clone()) {
            continue;
        }
        if info.name != EXTERNAL_CALL_TOOL && is_ts_identifier(&info.name) {
            externals.push(info.name);
        } else {
            log::debug!(
                "tool `{}` is not a valid script identifier; reachable via callTool only",
                info.name
            );
        }
    }
    Ok((externals, allowed))
}

// Reserved words that the registry's tool-name rule (`^[a-zA-Z0-9_-]{1,64}$`)
// would otherwise let through; exposing one as a global would not parse.
static RESERVED_WORDS: &[&str] = &[
    "async",
    "await",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "debugger",
    "default",
    "delete",
    "do",
    "else",
    "enum",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "function",
    "if",
    "import",
    "in",
    "instanceof",
    "let",
    "new",
    "null",
    "of",
    "return",
    "static",
    "super",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typeof",
    "undefined",
    "var",
    "void",
    "while",
    "with",
    "yield",
];

fn is_ts_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return false;
    }
    !RESERVED_WORDS.contains(&name)
}

/// Unwraps input that is exactly one fenced code block; anything else —
/// prose around a fence, multiple fences, no fence — is returned unchanged.
fn strip_code_fence(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return text;
    };
    let Some(rest) = rest.strip_suffix("```") else {
        return text;
    };
    // The first line is the info string ("ts"). A remaining ``` means the
    // input held more than one fence, so it is not a single block.
    match rest.split_once('\n') {
        Some((_, body)) if !body.contains("```") => body,
        _ => text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_code_fence_unwraps_single_fence() {
        assert_eq!(strip_code_fence("```ts\n1 + 2\n```"), "1 + 2\n");
        assert_eq!(
            strip_code_fence("  ```typescript\nlet a = 1;\na\n```  "),
            "let a = 1;\na\n"
        );
        assert_eq!(strip_code_fence("```\n1\n```"), "1\n");
    }

    #[test]
    fn strip_code_fence_leaves_other_text_unchanged() {
        assert_eq!(strip_code_fence("1 + 2"), "1 + 2");
        let prose = "Here is the code:\n```ts\n1\n```";
        assert_eq!(strip_code_fence(prose), prose);
        let two_fences = "```ts\n1\n```\nand\n```ts\n2\n```";
        assert_eq!(strip_code_fence(two_fences), two_fences);
        assert_eq!(strip_code_fence("``````"), "``````");
    }

    #[test]
    fn ts_identifier_accepts_plain_names_only() {
        assert!(is_ts_identifier("webSearch"));
        assert!(is_ts_identifier("_tool2"));
        assert!(!is_ts_identifier("web-search"));
        assert!(!is_ts_identifier("2fast"));
        assert!(!is_ts_identifier(""));
        assert!(!is_ts_identifier("delete"));
    }

    #[test]
    fn tool_externals_always_declares_call_tool() {
        let (externals, allowed) = tool_externals("").unwrap();
        assert_eq!(externals, vec![EXTERNAL_CALL_TOOL.to_string()]);
        assert!(allowed.is_empty());
    }

    #[test]
    fn tool_externals_rejects_bad_regex() {
        assert!(tool_externals("[unclosed").is_err());
    }
}
