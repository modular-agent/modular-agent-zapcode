use modular_agent_core::{
    Agent, AgentConfigs, AgentContext, AgentData, AgentError, AgentSpec, AgentStatus, AgentValue,
    AsAgent, ModularAgent, async_trait, modular_agent,
    tool::{Tool, ToolInfo, call_tool, register_tool, unregister_tool},
};
use zapcode_core::ResourceLimits;

use crate::runner::{ExternalHandler, run_zapcode};

static CATEGORY: &str = "Script/ZapCode";

static CONFIG_NAME: &str = "name";
static CONFIG_DESCRIPTION: &str = "description";
static CONFIG_PARAMETERS: &str = "parameters";
static CONFIG_SCRIPT: &str = "script";
static CONFIG_TIME_LIMIT_MS: &str = "time_limit_ms";
static CONFIG_MEMORY_LIMIT_MB: &str = "memory_limit_mb";

const DEFAULT_TIME_LIMIT_MS: i64 = 5000;
const DEFAULT_MEMORY_LIMIT_MB: i64 = 32;

/// Name of the external function scripts use to invoke other registered tools.
static CALL_TOOL_FN: &str = "callTool";

/// Defines an LLM tool implemented as a ZapCode (TypeScript subset) script.
///
/// While the agent is running, the tool is registered under `name` so LLM
/// agents whose `tools` patterns match it can call it. On each call the
/// tool's arguments are bound as script variables: every top-level argument
/// becomes a variable of the same name, and the whole argument object is also
/// available as `args` (the only way to reach arguments whose names are not
/// valid identifiers). The value of the last expression in the script is the
/// tool result. Script errors are returned to the calling LLM as an error
/// tool result, so the model sees the message and can retry.
///
/// Scripts can invoke other registered tools with
/// `await callTool(name, args)`, composing tools out of tools. There is no
/// recursion guard, and the time limit does not stop recursion — each nested
/// call gets its own time budget and time spent waiting on `callTool` is not
/// counted — so a tool that calls itself, directly or indirectly, hangs until
/// the flow is cancelled.
///
/// Scripts run in a sandbox with no filesystem, network, or environment
/// access, and execution is aborted when the time or memory limit is
/// exceeded. `console.log` output is written to the application log, tagged
/// with the tool name (output after the first `callTool` is not captured).
///
/// # Configuration
/// - `name`: Tool name; must match `^[a-zA-Z0-9_-]{1,64}$` (default: the
///   agent definition name)
/// - `description`: What the tool does and when to use it. Sent to the LLM —
///   a detailed description (3-4+ sentences) materially improves tool
///   selection
/// - `parameters`: JSON Schema describing the tool's arguments
/// - `script`: ZapCode script executed on each tool call
/// - `time_limit_ms`: Time limit in milliseconds for each stretch of script
///   execution between `callTool` calls; the clock restarts after every
///   `callTool`, so it does not bound the total call (default: 5000)
/// - `memory_limit_mb`: Script memory limit, in megabytes (default: 32)
///
/// # Example
/// With `parameters` declaring properties `a` and `b` and the script `a + b`,
/// a call with `{"a": 1, "b": 2}` returns `3`.
#[modular_agent(
    title = "ZapCode Tool",
    category = CATEGORY,
    string_config(name = CONFIG_NAME),
    text_config(name = CONFIG_DESCRIPTION),
    object_config(name = CONFIG_PARAMETERS),
    text_config(name = CONFIG_SCRIPT),
    integer_config(name = CONFIG_TIME_LIMIT_MS, default = DEFAULT_TIME_LIMIT_MS, detail),
    integer_config(name = CONFIG_MEMORY_LIMIT_MB, default = DEFAULT_MEMORY_LIMIT_MB, detail),
)]
struct ZapCodeToolAgent {
    data: AgentData,
    name: String,
}

impl ZapCodeToolAgent {
    /// Builds the tool from the current configs under the cached name.
    fn build_tool(&self) -> Result<ZapCodeScriptTool, AgentError> {
        let configs = self.configs()?;
        let description = configs.get_string_or_default(CONFIG_DESCRIPTION);
        let parameters = configs
            .get(CONFIG_PARAMETERS)
            .ok()
            .and_then(|v| serde_json::to_value(v).ok());
        Ok(ZapCodeScriptTool {
            info: ToolInfo::new(self.name.clone(), description, parameters),
            script: configs.get_string_or_default(CONFIG_SCRIPT),
            limits: limits_from_configs(configs),
        })
    }
}

#[async_trait]
impl AsAgent for ZapCodeToolAgent {
    fn new(ma: ModularAgent, id: String, spec: AgentSpec) -> Result<Self, AgentError> {
        let name = spec
            .configs
            .as_ref()
            .and_then(|c| c.get_string(CONFIG_NAME).ok())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| spec.def_name.clone());
        Ok(Self {
            data: AgentData::new(ma, id, spec),
            name,
        })
    }

    fn configs_changed(&mut self) -> Result<(), AgentError> {
        let new_name = {
            let configs = self.configs()?;
            configs
                .get_string(CONFIG_NAME)
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| self.def_name().to_string())
        };
        let old_name = std::mem::replace(&mut self.name, new_name);

        // Refresh the registration only while running; otherwise start() will
        // register the tool with the new values later.
        if self.data.status == AgentStatus::Start {
            warn_if_invalid_name(self.id(), &self.name);
            let tool = self.build_tool()?;
            refresh_registration(&old_name, tool);
        }

        Ok(())
    }

    async fn start(&mut self) -> Result<(), AgentError> {
        // Claude and OpenAI both require tool names to match
        // ^[a-zA-Z0-9_-]{1,64}$; an invalid name only fails later at API-call
        // time, so surface it early.
        warn_if_invalid_name(self.id(), &self.name);
        let tool = self.build_tool()?;
        register_tool(tool);
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), AgentError> {
        unregister_tool(&self.name);
        Ok(())
    }

    async fn process(
        &mut self,
        _ctx: AgentContext,
        _port: String,
        _value: AgentValue,
    ) -> Result<(), AgentError> {
        Ok(())
    }
}

/// Register first: for an in-place refresh this overwrites the entry
/// atomically, so concurrent lookups never hit a missing tool. The registry
/// is name-keyed and process-global, so on rename the old name must still be
/// removed explicitly or it would leak a stale entry that stop() (which
/// unregisters the new name) never cleans up.
fn refresh_registration(old_name: &str, tool: ZapCodeScriptTool) {
    let new_name = tool.info.name.clone();
    register_tool(tool);
    if old_name != new_name {
        unregister_tool(old_name);
    }
}

/// Local copy of core's tool-name rule (`^[a-zA-Z0-9_-]{1,64}$`); core does
/// not export it. Warn-only so a half-typed name never kills the agent.
fn is_valid_tool_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn warn_if_invalid_name(id: &str, name: &str) {
    if !is_valid_tool_name(name) {
        log::warn!(
            "ZapCodeToolAgent {} has invalid tool name {:?}; \
             tool names must match ^[a-zA-Z0-9_-]{{1,64}}$",
            id,
            name
        );
    }
}

fn limits_from_configs(configs: &AgentConfigs) -> ResourceLimits {
    let mut limits = ResourceLimits {
        // Every VM push counts toward the default 100k allocation cap, which
        // would trip far below what the exposed time/memory budgets allow;
        // lift it so time and memory are the effective limits.
        max_allocations: usize::MAX,
        ..ResourceLimits::default()
    };
    let time = configs.get_integer_or(CONFIG_TIME_LIMIT_MS, DEFAULT_TIME_LIMIT_MS);
    if time > 0 {
        limits.time_limit_ms = time as u64;
    }
    let memory = configs.get_integer_or(CONFIG_MEMORY_LIMIT_MB, DEFAULT_MEMORY_LIMIT_MB);
    if memory > 0 {
        limits.memory_limit_bytes = (memory as usize).saturating_mul(1024 * 1024);
    }
    limits
}

/// The registered tool: runs the script inline on each call, with the call's
/// arguments bound as variables and `callTool` bridged to the tool registry.
struct ZapCodeScriptTool {
    info: ToolInfo,
    script: String,
    limits: ResourceLimits,
}

#[async_trait]
impl Tool for ZapCodeScriptTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn call(&self, ctx: AgentContext, args: AgentValue) -> Result<AgentValue, AgentError> {
        let inputs = bind_args(&args);
        let mut handler = ToolCallHandler { ctx };
        let outcome = run_zapcode(
            self.script.clone(),
            inputs,
            vec![CALL_TOOL_FN.to_string()],
            self.limits.clone(),
            &mut handler,
        )
        .await?;

        if !outcome.console.is_empty() {
            log::info!("[tool {}] {}", self.info.name, outcome.console.trim_end());
        }

        Ok(outcome.value)
    }
}

/// Binds each top-level argument as a script variable, plus the whole value
/// as `args`. Non-identifier keys can't be script globals, so they stay
/// reachable only through `args`; a literal `args` argument wins over the
/// whole-object binding because duplicate input names must not be passed to
/// the VM.
fn bind_args(args: &AgentValue) -> Vec<(String, AgentValue)> {
    let mut inputs: Vec<(String, AgentValue)> = Vec::new();
    if let AgentValue::Object(map) = args {
        for (key, value) in map.iter() {
            if is_valid_identifier(key) {
                inputs.push((key.clone(), value.clone()));
            } else {
                log::debug!("tool argument {key:?} is not a valid identifier; use `args` instead");
            }
        }
    }
    if !inputs.iter().any(|(key, _)| key == "args") {
        inputs.push(("args".to_string(), args.clone()));
    }
    inputs
}

fn is_valid_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Bridges the `callTool(name, args)` external function to the process-global
/// tool registry.
struct ToolCallHandler {
    ctx: AgentContext,
}

#[async_trait]
impl ExternalHandler for ToolCallHandler {
    async fn call(
        &mut self,
        name: String,
        args: Vec<AgentValue>,
    ) -> Result<AgentValue, AgentError> {
        if name != CALL_TOOL_FN {
            return Err(AgentError::InvalidValue(format!(
                "ZapCode runtime error: external function `{name}` is not available here"
            )));
        }
        let mut args = args.into_iter();
        let tool_name = args
            .next()
            .and_then(|v| v.as_str().map(str::to_string))
            .ok_or_else(|| {
                AgentError::InvalidValue(
                    "callTool: first argument must be a tool name string".into(),
                )
            })?;
        let tool_args = args.next().unwrap_or(AgentValue::Unit);
        call_tool(self.ctx.clone(), &tool_name, tool_args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modular_agent_core::tool::get_tool;

    fn make_tool(name: &str, script: &str) -> ZapCodeScriptTool {
        ZapCodeScriptTool {
            info: ToolInfo::new(name, "test tool for the zapcode tool suite", None),
            script: script.to_string(),
            limits: ResourceLimits::default(),
        }
    }

    fn object_args(pairs: &[(&str, i64)]) -> AgentValue {
        let mut map = im::HashMap::new();
        for (k, v) in pairs {
            map.insert(k.to_string(), AgentValue::Integer(*v));
        }
        AgentValue::Object(map)
    }

    // The tool registry is process-global, so every test uses unique names.

    #[tokio::test]
    async fn registered_tool_is_callable_until_unregistered() {
        let name = "zapcode-tool-test-roundtrip";
        register_tool(make_tool(name, "a + b"));

        let result = call_tool(
            AgentContext::new(),
            name,
            object_args(&[("a", 1), ("b", 2)]),
        )
        .await
        .unwrap();
        assert!(matches!(result, AgentValue::Integer(3)));

        unregister_tool(name);
        let err = call_tool(AgentContext::new(), name, AgentValue::Unit)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn script_error_propagates_as_err() {
        let name = "zapcode-tool-test-error";
        register_tool(make_tool(name, "missingFn()"));

        // The Err lands in core's error_tool_result / is_error path, so the
        // calling LLM sees the message instead of the flow aborting.
        let err = call_tool(AgentContext::new(), name, AgentValue::Unit)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("ZapCode runtime error"), "{err}");

        unregister_tool(name);
    }

    #[tokio::test]
    async fn whole_argument_object_is_bound_as_args() {
        let name = "zapcode-tool-test-args";
        register_tool(make_tool(name, "args.x * 2"));

        let result = call_tool(AgentContext::new(), name, object_args(&[("x", 21)]))
            .await
            .unwrap();
        assert!(matches!(result, AgentValue::Integer(42)));

        unregister_tool(name);
    }

    #[tokio::test]
    async fn call_tool_external_composes_registered_tools() {
        let base = "zapcode-tool-test-compose-base";
        let outer = "zapcode-tool-test-compose-outer";
        register_tool(make_tool(base, "x + 1"));
        register_tool(make_tool(
            outer,
            "await callTool(\"zapcode-tool-test-compose-base\", {x: 41})",
        ));

        let result = call_tool(AgentContext::new(), outer, AgentValue::Unit)
            .await
            .unwrap();
        assert!(matches!(result, AgentValue::Integer(42)));

        unregister_tool(outer);
        unregister_tool(base);
    }

    #[tokio::test]
    async fn rename_refresh_cleans_up_the_old_name() {
        let old = "zapcode-tool-test-rename-old";
        let new = "zapcode-tool-test-rename-new";
        register_tool(make_tool(old, "1"));

        refresh_registration(old, make_tool(new, "2"));
        assert!(get_tool(old).is_none());
        assert!(get_tool(new).is_some());

        let result = call_tool(AgentContext::new(), new, AgentValue::Unit)
            .await
            .unwrap();
        assert!(matches!(result, AgentValue::Integer(2)));

        unregister_tool(new);
    }

    #[test]
    fn tool_name_validation_matches_core_rule() {
        assert!(is_valid_tool_name("web-search_2"));
        assert!(!is_valid_tool_name(""));
        assert!(!is_valid_tool_name("has space"));
        assert!(!is_valid_tool_name(&"x".repeat(65)));
    }
}
