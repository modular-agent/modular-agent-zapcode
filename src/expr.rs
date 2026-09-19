use modular_agent_core::{
    AsModule, Error, ModularAgent, Module, ModuleContext, ModuleData, ModuleOutput, ModuleSpec,
    Result, Value, async_trait, modular_agent, tool::call_tool,
};
use zapcode_core::ResourceLimits;

use crate::bridge::{ExternalHandler, run_zapcode};

static CATEGORY: &str = "Script/ZapCode";

static PORT_VALUE: &str = "value";

static CONFIG_EXPR: &str = "expr";
static CONFIG_SKIP_UNIT: &str = "skip_unit";
static CONFIG_TIME_LIMIT_MS: &str = "time_limit_ms";
static CONFIG_MEMORY_LIMIT_MB: &str = "memory_limit_mb";

static EXTERNAL_CALL_TOOL: &str = "callTool";

/// Evaluates a TypeScript-like expression or script over the input value.
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
///   (e.g. by a ZC Tool module) and returns its result; a failed tool call aborts
///   the script with an error
/// - `console.log` output is written to the application log, tagged with the
///   module id (output after the first `callTool` is not captured)
/// - Scripts are sandboxed: no filesystem, network, or environment access, and
///   no `import` / `require` / `eval`; execution stops with an error when a
///   resource limit is exceeded
///
/// # Ports
/// - Input `value`: Value passed to the script as the `value` variable
/// - Output `value`: Result of the last expression in the script
///
/// # Configuration
/// - `expr`: TypeScript-like expression or script to evaluate (text/multiline)
/// - `skip_unit`: When `true`, suppress output if the script results in
///   `undefined` or `null` (default: `false`)
/// - `time_limit_ms`: Time limit in milliseconds for each stretch of script
///   execution between `callTool` calls; the clock restarts after every
///   `callTool`, so it does not bound the total run (default: 5000)
/// - `memory_limit_mb`: Script memory limit in megabytes (default: 32)
///
/// # Example
/// With input `21` and expression `value * 2`, outputs `42`.
#[modular_agent(
    title = "ZC Expr",
    category = CATEGORY,
    inputs = [PORT_VALUE],
    outputs = [PORT_VALUE],
    text_config(name = CONFIG_EXPR),
    boolean_config(name = CONFIG_SKIP_UNIT, detail),
    integer_config(name = CONFIG_TIME_LIMIT_MS, default = 5000, detail),
    integer_config(name = CONFIG_MEMORY_LIMIT_MB, default = 32, detail),
)]
struct ZcExprModule {
    data: ModuleData,
}

#[async_trait]
impl AsModule for ZcExprModule {
    fn new(ma: ModularAgent, id: String, spec: ModuleSpec) -> Result<Self> {
        Ok(Self {
            data: ModuleData::new(ma, id, spec),
        })
    }

    async fn process(&mut self, ctx: ModuleContext, _port: String, value: Value) -> Result<()> {
        let config = self.configs()?;
        let script = config.get_string(CONFIG_EXPR)?;
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

        if skip_unit && matches!(outcome.value, Value::Unit) {
            return Ok(());
        }
        self.output(ctx, PORT_VALUE, outcome.value).await
    }
}

/// Bridges the script's `callTool(name, args)` external to the core tool
/// registry. `args` is optional on the guest side and defaults to unit.
struct ToolCallHandler {
    ctx: ModuleContext,
}

#[async_trait]
impl ExternalHandler for ToolCallHandler {
    async fn call(&mut self, name: String, mut args: Vec<Value>) -> Result<Value> {
        if name != EXTERNAL_CALL_TOOL {
            // Unreachable while callTool is the only declared external; kept as
            // a guard so a future externals change cannot silently misroute.
            return Err(Error::InvalidValue(format!(
                "ZapCode runtime error: external function `{name}` is not available here"
            )));
        }
        let tool_name = match args.first() {
            Some(Value::String(s)) => s.to_string(),
            _ => {
                return Err(Error::InvalidValue(
                    "callTool: first argument must be a tool name string".into(),
                ));
            }
        };
        let tool_args = if args.len() > 1 {
            args.swap_remove(1)
        } else {
            Value::Unit
        };
        call_tool(self.ctx.clone(), &tool_name, tool_args).await
    }
}
