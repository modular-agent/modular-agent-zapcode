use modular_agent_core::{
    Agent, AgentContext, AgentData, AgentError, AgentOutput, AgentSpec, AgentValue, AsAgent,
    ModularAgent, async_trait, modular_agent, tool::call_tool,
};
use zapcode_core::ResourceLimits;

use crate::runner::{ExternalHandler, run_zapcode};

static CATEGORY: &str = "Script/ZapCode";

static PORT_VALUE: &str = "value";

static CONFIG_SCRIPT: &str = "script";
static CONFIG_SKIP_UNIT: &str = "skip_unit";
static CONFIG_TIME_LIMIT_MS: &str = "time_limit_ms";
static CONFIG_MEMORY_LIMIT_MB: &str = "memory_limit_mb";

static EXTERNAL_CALL_TOOL: &str = "callTool";

/// ZapCode Script agent for executing TypeScript-like scripts.
///
/// Uses [ZapCode](https://github.com/TheUncharted/zapcode), a Rust-native
/// sandboxed TypeScript-subset interpreter, to run user-provided scripts with
/// the input value as a parameter.
///
/// - Scripts receive input as the variable `value`
/// - The last expression's value becomes the output (`undefined` and `null`
///   both become a unit value)
/// - Scripts are compiled fresh on each invocation; an empty script does nothing
/// - `await callTool(name, args)` calls a tool registered in this application
///   (e.g. by a Tool agent) and returns its result; a failed tool call aborts
///   the script with an error
/// - `console.log` output is written to the application log, tagged with the
///   agent id (output after the first `callTool` is not captured)
/// - Scripts are sandboxed: no filesystem, network, or environment access, and
///   no `import` / `require` / `eval`; execution stops with an error when a
///   resource limit is exceeded
///
/// # Ports
/// - Input `value`: Value passed to the script as the `value` variable
/// - Output `value`: Result of the last expression in the script
///
/// # Configuration
/// - `script`: TypeScript-like script to execute (text/multiline)
/// - `skip_unit`: When `true`, suppress output if the script results in
///   `undefined` or `null` (default: `false`)
/// - `time_limit_ms`: Time limit in milliseconds for each stretch of script
///   execution between `callTool` calls; the clock restarts after every
///   `callTool`, so it does not bound the total run (default: 5000)
/// - `memory_limit_mb`: Script memory limit in megabytes (default: 32)
///
/// # Example
/// With input `21` and script `value * 2`, outputs `42`.
#[modular_agent(
    title = "ZapCode Script",
    category = CATEGORY,
    inputs = [PORT_VALUE],
    outputs = [PORT_VALUE],
    text_config(name = CONFIG_SCRIPT),
    boolean_config(name = CONFIG_SKIP_UNIT),
    integer_config(name = CONFIG_TIME_LIMIT_MS, default = 5000, detail),
    integer_config(name = CONFIG_MEMORY_LIMIT_MB, default = 32, detail),
)]
struct ZapCodeScriptAgent {
    data: AgentData,
}

#[async_trait]
impl AsAgent for ZapCodeScriptAgent {
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
        let script = config.get_string(CONFIG_SCRIPT)?;
        if script.is_empty() {
            return Ok(());
        }
        let skip_unit = config.get_bool_or_default(CONFIG_SKIP_UNIT);
        let limits = ResourceLimits {
            time_limit_ms: config.get_integer_or(CONFIG_TIME_LIMIT_MS, 5000).max(1) as u64,
            memory_limit_bytes: config.get_integer_or(CONFIG_MEMORY_LIMIT_MB, 32).max(1) as usize
                * 1024
                * 1024,
            // max_allocations counts VM stack pushes — an instruction-rate
            // proxy, not memory — and the 100k default halts ordinary loops
            // within milliseconds. The wall-clock time limit is the budget.
            max_allocations: usize::MAX,
            ..ResourceLimits::default()
        };

        let mut handler = ToolCallHandler { ctx: ctx.clone() };
        let outcome = run_zapcode(
            script,
            vec![(PORT_VALUE.to_string(), value)],
            vec![EXTERNAL_CALL_TOOL.to_string()],
            limits,
            &mut handler,
        )
        .await?;

        if !outcome.console.is_empty() {
            log::info!("[{}] {}", self.id(), outcome.console.trim_end_matches('\n'));
        }

        if skip_unit && matches!(outcome.value, AgentValue::Unit) {
            return Ok(());
        }
        self.output(ctx, PORT_VALUE, outcome.value).await
    }
}

/// Bridges the script's `callTool(name, args)` external to the core tool
/// registry. `args` is optional on the guest side and defaults to unit.
struct ToolCallHandler {
    ctx: AgentContext,
}

#[async_trait]
impl ExternalHandler for ToolCallHandler {
    async fn call(
        &mut self,
        name: String,
        mut args: Vec<AgentValue>,
    ) -> Result<AgentValue, AgentError> {
        if name != EXTERNAL_CALL_TOOL {
            // Unreachable while callTool is the only declared external; kept as
            // a guard so a future externals change cannot silently misroute.
            return Err(AgentError::InvalidValue(format!(
                "ZapCode runtime error: external function `{name}` is not available here"
            )));
        }
        let tool_name = match args.first() {
            Some(AgentValue::String(s)) => s.to_string(),
            _ => {
                return Err(AgentError::InvalidValue(
                    "callTool: first argument must be a tool name string".into(),
                ));
            }
        };
        let tool_args = if args.len() > 1 {
            args.swap_remove(1)
        } else {
            AgentValue::Unit
        };
        call_tool(self.ctx.clone(), &tool_name, tool_args).await
    }
}
