use std::collections::HashMap;

use modular_agent_core::{
    AsModule, Error, ModularAgent, Module, ModuleConfigSpec, ModuleConfigSpecs, ModuleConfigs,
    ModuleContext, ModuleData, ModuleOutput, ModuleSpec, Result, Value, async_trait, modular_agent,
    tool::call_tool,
};
use zapcode_core::{ResourceLimits, VmState, ZapcodeRun};

use crate::bridge::{ExternalHandler, map_zapcode_error, run_zapcode};
use crate::value::zapcode_to_value;

static CATEGORY: &str = "Script/ZapCode";

static PORT_VALUE: &str = "value";

static CONFIG_SCRIPT: &str = "script";
static CONFIG_TIME_LIMIT_MS: &str = "time_limit_ms";
static CONFIG_MEMORY_LIMIT_MB: &str = "memory_limit_mb";

// Configs owned by the module itself; a script may not declare these names.
static BASE_CONFIGS: &[&str] = &[CONFIG_SCRIPT, CONFIG_TIME_LIMIT_MS, CONFIG_MEMORY_LIMIT_MB];

static EXTERNALS: &[&str] = &[
    "emit",
    "getConfig",
    "getState",
    "setState",
    "log",
    "callTool",
];

static INPUT_PORT_VAR: &str = "__port__";
static INPUT_VALUE_VAR: &str = "__value__";

// The describe run only evaluates a declaration object; a short wall-clock
// budget doubles as enforcement that top-level code stays side-effect free.
const DESCRIBE_TIME_LIMIT_MS: u64 = 500;

const DEFAULT_TIME_LIMIT_MS: i64 = 5000;
const DEFAULT_MEMORY_LIMIT_MB: i64 = 32;

/// Generic node whose ports, configs, and behavior are defined by a ZapCode
/// (sandboxed TypeScript subset) script.
///
/// The script declares the node's shape in a top-level `MODULE` object and
/// implements its behavior in an `onInput(port, value)` function (the name
/// `process` is reserved by the sandbox):
///
/// ```ts
/// const MODULE = {
///   inputs: ["value", "reset"],          // default: ["value"]
///   outputs: ["avg"],                    // default: ["value"]
///   configs: {
///     window: { type: "integer", value: 5, title: "Window" },
///   },
///   api: 1,
/// };
/// function onInput(port, value) {
///   if (port === "reset") { setState("samples", []); return; }
///   const n = getConfig("window");
///   const samples = (getState("samples") ?? []).concat([value]).slice(-n);
///   setState("samples", samples);
///   emit("avg", samples.reduce((a, b) => a + b) / samples.length);
/// }
/// ```
///
/// When the script changes, `MODULE` is re-evaluated (with no host functions
/// and a short time budget, so top-level code must be free of side effects)
/// and the node's ports and config fields update immediately. A broken script
/// does not kill the node: it keeps its last valid ports and configs, and the
/// error is reported. `name`, `title`, `category`, and `description` keys in
/// `MODULE` are accepted and ignored. Declared config fields appear in the
/// inspector sorted by name.
///
/// Each input triggers `onInput(port, value)`. Inside it, these host
/// functions are available:
///
/// - `emit(port, value)`: Send a value to a declared output port. Emits are
///   collected and delivered after the script finishes; a failed run delivers
///   nothing.
/// - `getConfig(name)`: Read a config value (`null` when unset).
/// - `getState(key)` / `setState(key, value)`: Per-node state that persists
///   across runs. It lives in memory only and is not saved with the patch.
/// - `log(message)`: Write a message to the application log.
/// - `callTool(name, args)`: Call a registered LLM tool and return its result.
///
/// The sandbox has no filesystem, network, or environment access; `import`,
/// `require`, and `eval` are unavailable.
///
/// # Ports
/// - Input `value`: Default input port; replaced by the `inputs` declared in `MODULE`.
/// - Output `value`: Default output port; replaced by the `outputs` declared in `MODULE`.
///
/// # Configuration
/// - `script`: ZapCode script declaring the `MODULE` object and the `onInput` function.
///   Script-declared configs appear as additional fields. An empty script leaves
///   the node inert with the default ports.
/// - `time_limit_ms`: Time limit in milliseconds for each stretch of script
///   execution between host-function calls; the clock restarts after every
///   `emit` / `getConfig` / `getState` / `setState` / `log` / `callTool`, so
///   it does not bound the total duration of an `onInput` run (default: 5000)
/// - `memory_limit_mb`: Memory budget for one `onInput` run in megabytes (default: 32)
///
/// # Example
/// With the moving-average script above, sending `1`, `2`, `3` to `value`
/// emits `1`, `1.5`, `2` on `avg`; sending anything to `reset` clears the window.
#[modular_agent(
    title = "ZC Script",
    category = CATEGORY,
    inputs = [PORT_VALUE],
    outputs = [PORT_VALUE],
    text_config(name = CONFIG_SCRIPT, detail),
    integer_config(name = CONFIG_TIME_LIMIT_MS, default = 5000, detail),
    integer_config(name = CONFIG_MEMORY_LIMIT_MB, default = 32, detail),
)]
struct ZcScriptModule {
    data: ModuleData,

    // Outputs of the last successful describe; emit() validates against these.
    outputs: Vec<String>,

    // Script whose describe last succeeded. Lets configs_changed skip the
    // describe run when only config values changed, and retry (and re-report)
    // when the script itself is still broken.
    valid_script: Option<String>,

    state: HashMap<String, Value>,
}

#[async_trait]
impl AsModule for ZcScriptModule {
    fn new(ma: ModularAgent, id: String, mut spec: ModuleSpec) -> Result<Self> {
        let script = spec
            .configs
            .as_ref()
            .map(|cfg| cfg.get_string_or_default(CONFIG_SCRIPT))
            .unwrap_or_default();

        // A broken script must not kill the node at load time: keep the spec
        // as reconciled (default value/value ports) and report on next edit.
        let (outputs, valid_script) = match describe_and_apply(&script, &mut spec) {
            Ok(decl) => (decl.outputs, Some(script)),
            Err(e) => {
                log::warn!("[{id}] ZC Script describe failed: {e}");
                // ModuleData::new strips every `_`-prefixed config, so the
                // saved values of script-declared configs (renamed by
                // reconcile_spec) would be gone for good once the user fixes
                // the script. Rename them back so a later successful describe
                // still finds them.
                restore_stale_configs(&mut spec);
                (
                    spec.outputs
                        .clone()
                        .unwrap_or_else(|| vec![PORT_VALUE.to_string()]),
                    None,
                )
            }
        };

        Ok(Self {
            data: ModuleData::new(ma, id, spec),
            outputs,
            valid_script,
            state: HashMap::new(),
        })
    }

    fn configs_changed(&mut self) -> Result<()> {
        let script = self.configs()?.get_string_or_default(CONFIG_SCRIPT);
        if self.valid_script.as_deref() == Some(script.as_str()) {
            return Ok(());
        }

        // On failure the spec is untouched (describe_and_apply mutates only
        // after every fallible step), so the node keeps its last valid shape
        // and the error surfaces to the user.
        let decl = describe_and_apply(&script, &mut self.data.spec)?;
        self.outputs = decl.outputs;
        self.valid_script = Some(script);
        self.emit_module_spec_updated();
        Ok(())
    }

    async fn process(&mut self, ctx: ModuleContext, port: String, value: Value) -> Result<()> {
        let (script, limits, configs) = {
            let config = self.configs()?;
            (
                config.get_string_or_default(CONFIG_SCRIPT),
                limits_from_configs(config),
                config.clone(),
            )
        };
        if script.trim().is_empty() {
            return Ok(());
        }

        // `await` unwraps the promise an `async function onInput` returns and
        // is a no-op for a plain function, so both declarations work.
        let source = format!("{script}\nawait onInput({INPUT_PORT_VAR}, {INPUT_VALUE_VAR})");
        let inputs = vec![
            (INPUT_PORT_VAR.to_string(), Value::string(port)),
            (INPUT_VALUE_VAR.to_string(), value),
        ];
        let externals = EXTERNALS.iter().map(|s| s.to_string()).collect();

        // The handler borrows the state map in place: core may cancel process()
        // by dropping this future at an await point, and a moved-out map would
        // be lost with the handler. State mutations made before a failure are
        // kept either way: the script observed them, so dropping them would
        // fork guest and host views of the state.
        let mut handler = ProcessHandler {
            ctx: ctx.clone(),
            module_id: self.id().to_string(),
            configs,
            outputs: self.outputs.clone(),
            state: &mut self.state,
            emits: Vec::new(),
        };
        let result = run_zapcode(source, inputs, externals, limits, &mut handler).await;
        // Destructuring ends the &mut self.state borrow so self.output can run.
        let ProcessHandler { emits, .. } = handler;
        let outcome = result?;

        if !outcome.console.is_empty() {
            log::info!("[{}] {}", self.id(), outcome.console.trim_end());
        }
        for (port, value) in emits {
            self.output(ctx.clone(), port, value).await?;
        }
        Ok(())
    }
}

fn limits_from_configs(config: &ModuleConfigs) -> ResourceLimits {
    let time_limit_ms = config
        .get_integer_or(CONFIG_TIME_LIMIT_MS, DEFAULT_TIME_LIMIT_MS)
        .max(1) as u64;
    let memory_limit_mb = config
        .get_integer_or(CONFIG_MEMORY_LIMIT_MB, DEFAULT_MEMORY_LIMIT_MB)
        .max(1) as usize;
    ResourceLimits {
        time_limit_ms,
        memory_limit_bytes: memory_limit_mb * 1024 * 1024,
        // max_allocations counts VM stack pushes — an instruction-rate proxy,
        // not memory — and the 100k default halts ordinary loops within
        // milliseconds. Time and memory are the effective budgets.
        max_allocations: usize::MAX,
        ..ResourceLimits::default()
    }
}

// ---------------------------------------------------------------------------
// Describe path: evaluate `MODULE` and rebuild the spec from it
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Declaration {
    inputs: Vec<String>,
    outputs: Vec<String>,
    // Sorted by name: the Object -> im::HashMap conversion loses the script's
    // declaration order, so sorting is the only deterministic choice.
    configs: Vec<(String, ModuleConfigSpec)>,
}

impl Declaration {
    fn default_ports() -> Self {
        Self {
            inputs: vec![PORT_VALUE.to_string()],
            outputs: vec![PORT_VALUE.to_string()],
            configs: Vec::new(),
        }
    }
}

/// Moves `_`-prefixed config values (reconcile_spec's stale renames) back to
/// their plain names. An existing plain-name value wins, matching the
/// precedence `apply_declaration` uses when reading the fallback.
fn restore_stale_configs(spec: &mut ModuleSpec) {
    let Some(configs) = spec.configs.as_mut() else {
        return;
    };
    let stale: Vec<String> = configs
        .keys()
        .filter(|k| k.starts_with('_'))
        .cloned()
        .collect();
    for key in stale {
        if let Some(value) = configs.remove(&key) {
            let name = key[1..].to_string();
            if !name.is_empty() && !configs.contains_key(&name) {
                configs.set(name, value);
            }
        }
    }
}

fn describe_and_apply(script: &str, spec: &mut ModuleSpec) -> Result<Declaration> {
    let decl = describe(script)?;
    apply_declaration(spec, &decl)?;
    Ok(decl)
}

/// Evaluates `script + "\nMODULE"` and parses the resulting declaration.
/// An empty script declares nothing and yields the default ports.
fn describe(script: &str) -> Result<Declaration> {
    if script.trim().is_empty() {
        return Ok(Declaration::default_ports());
    }
    let value = run_describe_script(script)?;
    parse_declaration(&value)
}

// Runs the VM inline: new()/configs_changed() are synchronous lifecycle hooks,
// and with no external functions the run cannot suspend, so the async bridge
// in runner.rs is unnecessary here.
fn run_describe_script(script: &str) -> Result<Value> {
    let source = format!("{script}\nMODULE");
    let limits = ResourceLimits {
        time_limit_ms: DESCRIBE_TIME_LIMIT_MS,
        ..ResourceLimits::default()
    };
    let runner =
        ZapcodeRun::new(source, Vec::new(), Vec::new(), limits).map_err(map_zapcode_error)?;
    let result = runner.run(Vec::new()).map_err(map_zapcode_error)?;
    match result.state {
        VmState::Complete(v) => zapcode_to_value(v),
        // Unreachable with no declared externals; calling an unknown function
        // is a runtime error, not a suspension.
        VmState::Suspended { function_name, .. } => Err(Error::InvalidValue(format!(
            "ZapCode runtime error: external function `{function_name}` is not available \
             while evaluating MODULE"
        ))),
    }
}

fn parse_declaration(value: &Value) -> Result<Declaration> {
    // An unknown global evaluates to `undefined` rather than erroring, so a
    // script without a declaration lands here as Unit.
    if matches!(value, Value::Unit) {
        return Err(Error::InvalidConfig(
            "the script must declare a top-level MODULE object".to_string(),
        ));
    }
    let Value::Object(map) = value else {
        return Err(Error::InvalidConfig("MODULE must be an object".to_string()));
    };

    let inputs = parse_port_list(map.get("inputs"), "inputs")?;
    let outputs = parse_port_list(map.get("outputs"), "outputs")?;

    let mut configs = Vec::new();
    if let Some(configs_value) = map.get("configs") {
        let Value::Object(config_map) = configs_value else {
            return Err(Error::InvalidConfig(
                "MODULE.configs must be an object".to_string(),
            ));
        };
        let mut entries: Vec<(&String, &Value)> = config_map.iter().collect();
        entries.sort_by_key(|(name, _)| *name);
        for (name, entry) in entries {
            validate_config_name(name)?;
            configs.push((name.clone(), parse_config_decl(name, entry)?));
        }
    }

    // Remaining keys (name, title, category, description, api, ...) are
    // accepted and ignored for compatibility with the file-based module format.
    Ok(Declaration {
        inputs,
        outputs,
        configs,
    })
}

fn parse_port_list(value: Option<&Value>, key: &str) -> Result<Vec<String>> {
    let Some(value) = value else {
        return Ok(vec![PORT_VALUE.to_string()]);
    };
    let Value::Array(items) = value else {
        return Err(Error::InvalidConfig(format!(
            "MODULE.{key} must be an array of strings"
        )));
    };
    items
        .iter()
        .map(|item| {
            item.as_str()
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "MODULE.{key} must be an array of non-empty strings"
                    ))
                })
        })
        .collect()
}

fn validate_config_name(name: &str) -> Result<()> {
    if name.is_empty() || name.starts_with('_') {
        return Err(Error::InvalidConfig(format!(
            "MODULE.configs name `{name}` is invalid: it must be non-empty and must not \
             start with `_`"
        )));
    }
    if BASE_CONFIGS.contains(&name) {
        return Err(Error::InvalidConfig(format!(
            "MODULE.configs name `{name}` conflicts with a built-in config"
        )));
    }
    Ok(())
}

fn parse_config_decl(name: &str, entry: &Value) -> Result<ModuleConfigSpec> {
    let Value::Object(map) = entry else {
        return Err(Error::InvalidConfig(format!(
            "MODULE.configs.{name} must be an object"
        )));
    };

    let declared_type = match map.get("type") {
        Some(t) => Some(t.as_str().map(str::to_string).ok_or_else(|| {
            Error::InvalidConfig(format!("MODULE.configs.{name}.type must be a string"))
        })?),
        None => None,
    };
    let declared_value = map.get("value").cloned();

    let (type_, value) = match (declared_type, declared_value) {
        (Some(t), Some(v)) => (t, v),
        (Some(t), None) => {
            let v = default_for_type(&t);
            (t, v)
        }
        (None, Some(v)) => (infer_type(&v).to_string(), v),
        (None, None) => ("string".to_string(), Value::string_default()),
    };

    let mut config_spec = ModuleConfigSpec::new(value, &type_);
    config_spec.title = map
        .get("title")
        .and_then(|t| t.as_str())
        .map(str::to_string);
    config_spec.description = map
        .get("description")
        .and_then(|d| d.as_str())
        .map(str::to_string);
    Ok(config_spec)
}

fn default_for_type(type_: &str) -> Value {
    match type_ {
        "integer" => Value::integer(0),
        "number" => Value::number(0.0),
        "boolean" => Value::boolean(false),
        "object" => Value::object(im::HashMap::new()),
        "array" => Value::array(im::Vector::new()),
        _ => Value::string_default(),
    }
}

fn infer_type(value: &Value) -> &'static str {
    match value {
        Value::Boolean(_) => "boolean",
        Value::Integer(_) => "integer",
        Value::Number(_) => "number",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
        _ => "string",
    }
}

/// Rebuilds `spec` ports/configs/config_specs from a declaration, keeping the
/// built-in configs and the stored values of script-declared configs.
///
/// Mutates `spec` only after every fallible step, so an error leaves it intact.
fn apply_declaration(spec: &mut ModuleSpec, decl: &Declaration) -> Result<()> {
    let get_base_spec = |name: &str| -> Result<ModuleConfigSpec> {
        spec.config_specs
            .as_ref()
            .and_then(|cs| cs.get(name))
            .cloned()
            .ok_or_else(|| Error::InvalidConfig(format!("config {name} must be present")))
    };
    let script_spec = get_base_spec(CONFIG_SCRIPT)?;
    let time_spec = get_base_spec(CONFIG_TIME_LIMIT_MS)?;
    let memory_spec = get_base_spec(CONFIG_MEMORY_LIMIT_MB)?;

    let old = spec.configs.clone().unwrap_or_default();
    let mut configs = ModuleConfigs::new();
    let mut config_specs = ModuleConfigSpecs::default();

    // Order determines the inspector layout: script on top, declared fields
    // next, resource limits last (they carry the `detail` flag).
    configs.set(
        CONFIG_SCRIPT.to_string(),
        Value::string(old.get_string_or_default(CONFIG_SCRIPT)),
    );
    config_specs.insert(CONFIG_SCRIPT.to_string(), script_spec);

    for (name, config_spec) in &decl.configs {
        // `ModuleDefinition::reconcile_spec` moves configs the definition does
        // not declare - which includes every script-declared one - to a
        // `_`-prefixed key when a patch is loaded. Fall back to it so saved
        // values survive a reload.
        let stale_name = format!("_{name}");
        let value = if old.contains_key(name) {
            old.get(name)?.clone()
        } else if old.contains_key(&stale_name) {
            old.get(&stale_name)?.clone()
        } else {
            config_spec.value.clone()
        };
        configs.set(name.clone(), value);
        config_specs.insert(name.clone(), config_spec.clone());
    }

    for (name, config_spec) in [
        (CONFIG_TIME_LIMIT_MS, time_spec),
        (CONFIG_MEMORY_LIMIT_MB, memory_spec),
    ] {
        let default = config_spec.value.as_i64().unwrap_or_default();
        configs.set(
            name.to_string(),
            Value::integer(old.get_integer_or(name, default)),
        );
        config_specs.insert(name.to_string(), config_spec);
    }

    spec.configs = Some(configs);
    spec.config_specs = Some(config_specs);
    spec.inputs = Some(decl.inputs.clone());
    spec.outputs = Some(decl.outputs.clone());
    Ok(())
}

// ---------------------------------------------------------------------------
// Process path: host functions available to `process(port, value)`
// ---------------------------------------------------------------------------

struct ProcessHandler<'a> {
    ctx: ModuleContext,
    module_id: String,
    configs: ModuleConfigs,
    outputs: Vec<String>,
    state: &'a mut HashMap<String, Value>,
    emits: Vec<(String, Value)>,
}

impl ProcessHandler<'_> {
    fn string_arg(args: &[Value], index: usize, usage: &str) -> Result<String> {
        args.get(index)
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or_else(|| Error::InvalidValue(format!("ZapCode runtime error: {usage}")))
    }

    fn value_arg(args: &[Value], index: usize) -> Value {
        args.get(index).cloned().unwrap_or(Value::Unit)
    }
}

#[async_trait]
impl ExternalHandler for ProcessHandler<'_> {
    async fn call(&mut self, name: String, args: Vec<Value>) -> Result<Value> {
        match name.as_str() {
            "emit" => {
                let port = Self::string_arg(&args, 0, "emit expects (port, value)")?;
                if !self.outputs.contains(&port) {
                    return Err(Error::InvalidValue(format!(
                        "ZapCode runtime error: emit to undeclared output port `{port}` \
                         (declared outputs: {})",
                        self.outputs.join(", ")
                    )));
                }
                self.emits.push((port, Self::value_arg(&args, 1)));
                Ok(Value::Unit)
            }
            "getConfig" => {
                let key = Self::string_arg(&args, 0, "getConfig expects (name)")?;
                Ok(self.configs.get(&key).cloned().unwrap_or(Value::Unit))
            }
            "getState" => {
                let key = Self::string_arg(&args, 0, "getState expects (key)")?;
                Ok(self.state.get(&key).cloned().unwrap_or(Value::Unit))
            }
            "setState" => {
                let key = Self::string_arg(&args, 0, "setState expects (key, value)")?;
                self.state.insert(key, Self::value_arg(&args, 1));
                Ok(Value::Unit)
            }
            "log" => {
                let message = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    Some(other) => other.to_json().to_string(),
                    None => String::new(),
                };
                log::info!("[{}] {message}", self.module_id);
                Ok(Value::Unit)
            }
            "callTool" => {
                let tool_name = Self::string_arg(&args, 0, "callTool expects (name, args)")?;
                call_tool(self.ctx.clone(), &tool_name, Self::value_arg(&args, 1)).await
            }
            // Unreachable: the VM only suspends on functions in EXTERNALS.
            other => Err(Error::InvalidValue(format!(
                "ZapCode runtime error: unknown external function `{other}`"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config_specs() -> ModuleConfigSpecs {
        let mut specs = ModuleConfigSpecs::default();
        specs.insert(
            CONFIG_SCRIPT.to_string(),
            ModuleConfigSpec::new(Value::string_default(), "text"),
        );
        specs.insert(
            CONFIG_TIME_LIMIT_MS.to_string(),
            ModuleConfigSpec::new(Value::integer(DEFAULT_TIME_LIMIT_MS), "integer"),
        );
        specs.insert(
            CONFIG_MEMORY_LIMIT_MB.to_string(),
            ModuleConfigSpec::new(Value::integer(DEFAULT_MEMORY_LIMIT_MB), "integer"),
        );
        specs
    }

    fn spec_with_configs(configs: ModuleConfigs) -> ModuleSpec {
        ModuleSpec {
            id: "test".to_string(),
            def_name: "test_def".to_string(),
            inputs: Some(vec![PORT_VALUE.to_string()]),
            outputs: Some(vec![PORT_VALUE.to_string()]),
            configs: Some(configs),
            config_specs: Some(base_config_specs()),
            ..ModuleSpec::default()
        }
    }

    static MOVING_AVERAGE_SCRIPT: &str = r#"
const MODULE = {
  inputs: ["value", "reset"],
  outputs: ["avg"],
  configs: {
    window: { type: "integer", value: 5, title: "Window" },
  },
  api: 1,
};
"#;

    #[test]
    fn describe_extracts_module_declaration() {
        let decl = describe(MOVING_AVERAGE_SCRIPT).unwrap();
        assert_eq!(decl.inputs, vec!["value", "reset"]);
        assert_eq!(decl.outputs, vec!["avg"]);
        assert_eq!(decl.configs.len(), 1);
        let (name, spec) = &decl.configs[0];
        assert_eq!(name, "window");
        assert_eq!(spec.type_.as_deref(), Some("integer"));
        assert!(matches!(spec.value, Value::Integer(5)));
        assert_eq!(spec.title.as_deref(), Some("Window"));
    }

    #[test]
    fn describe_defaults_when_keys_missing() {
        // name/title/category/description are accepted and ignored.
        let decl = describe(r#"const MODULE = { api: 1, name: "x", title: "y" };"#).unwrap();
        assert_eq!(decl.inputs, vec![PORT_VALUE]);
        assert_eq!(decl.outputs, vec![PORT_VALUE]);
        assert!(decl.configs.is_empty());
    }

    #[test]
    fn describe_empty_script_gives_default_ports() {
        let decl = describe("  \n ").unwrap();
        assert_eq!(decl.inputs, vec![PORT_VALUE]);
        assert_eq!(decl.outputs, vec![PORT_VALUE]);
        assert!(decl.configs.is_empty());
    }

    #[test]
    fn describe_broken_script_errors() {
        let err = describe("const = ;").unwrap_err();
        assert!(err.to_string().contains("ZapCode compile error"), "{err}");

        // A script without an MODULE declaration is a describe failure too
        // (an unknown global evaluates to `undefined`, not a runtime error).
        let err = describe("const x = 1;").unwrap_err();
        assert!(err.to_string().contains("must declare"), "{err}");

        let err = describe("const MODULE = 42;").unwrap_err();
        assert!(
            err.to_string().contains("MODULE must be an object"),
            "{err}"
        );
    }

    #[test]
    fn failed_describe_leaves_spec_untouched() {
        let mut configs = ModuleConfigs::new();
        configs.set(CONFIG_SCRIPT.to_string(), Value::string("const = ;"));
        configs.set("window".to_string(), Value::integer(7));
        let mut spec = spec_with_configs(configs);
        // A previously valid shape that a broken re-describe must not disturb.
        spec.inputs = Some(vec!["value".to_string(), "reset".to_string()]);
        spec.outputs = Some(vec!["avg".to_string()]);
        let before = serde_json::to_value(&spec).unwrap();

        let err = describe_and_apply("const = ;", &mut spec).unwrap_err();
        assert!(err.to_string().contains("ZapCode compile error"), "{err}");
        assert_eq!(serde_json::to_value(&spec).unwrap(), before);

        // A script that runs but declares no MODULE fails after the VM run;
        // the spec must survive that later failure too.
        let err = describe_and_apply("const x = 1;", &mut spec).unwrap_err();
        assert!(err.to_string().contains("must declare"), "{err}");
        assert_eq!(serde_json::to_value(&spec).unwrap(), before);
    }

    #[test]
    fn restore_stale_configs_renames_back_without_clobbering() {
        let mut configs = ModuleConfigs::new();
        configs.set("_window".to_string(), Value::integer(7));
        configs.set("kept".to_string(), Value::integer(1));
        configs.set("_kept".to_string(), Value::integer(2));
        let mut spec = spec_with_configs(configs);

        restore_stale_configs(&mut spec);

        let configs = spec.configs.as_ref().unwrap();
        assert!(matches!(configs.get("window").unwrap(), Value::Integer(7)));
        assert!(!configs.contains_key("_window"));
        // An existing plain-name value wins, matching apply_declaration.
        assert!(matches!(configs.get("kept").unwrap(), Value::Integer(1)));
        assert!(!configs.contains_key("_kept"));
    }

    // The full doc-comment example, end to end. Guards against silent zapcode
    // gaps: v1.5.3 does not implement the spread operator, so an earlier
    // `[...prev, value]` version nested the array and averaged to NaN.
    #[tokio::test]
    async fn doc_example_moving_average_emits_averages() {
        let script = r#"
const MODULE = {
  inputs: ["value", "reset"],
  outputs: ["avg"],
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
"#;
        let mut configs = ModuleConfigs::new();
        configs.set("window".to_string(), Value::integer(5));
        let mut state = HashMap::new();
        let mut handler = process_handler(&["avg"], configs, &mut state);

        run_process(&mut handler, script, Value::number(10.0))
            .await
            .unwrap();
        run_process(&mut handler, script, Value::number(20.0))
            .await
            .unwrap();

        let avgs: Vec<f64> = handler
            .emits
            .iter()
            .map(|(port, v)| {
                assert_eq!(port, "avg");
                v.as_f64().expect("avg must be a number")
            })
            .collect();
        assert_eq!(avgs, vec![10.0, 15.0]);
    }

    #[test]
    fn declared_configs_are_sorted_by_name() {
        let decl = describe(
            r#"const MODULE = { configs: { zebra: { value: 1 }, apple: { value: 2 } } };"#,
        )
        .unwrap();
        let names: Vec<&str> = decl.configs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["apple", "zebra"]);
    }

    #[test]
    fn declared_config_name_may_not_shadow_builtin() {
        let err =
            describe(r#"const MODULE = { configs: { script: { value: "x" } } };"#).unwrap_err();
        assert!(err.to_string().contains("built-in config"), "{err}");
    }

    #[test]
    fn apply_declaration_rebuilds_spec_and_reads_stale_fallback() {
        let mut configs = ModuleConfigs::new();
        configs.set(
            CONFIG_SCRIPT.to_string(),
            Value::string(MOVING_AVERAGE_SCRIPT),
        );
        configs.set(CONFIG_TIME_LIMIT_MS.to_string(), Value::integer(1234));
        // Simulates reconcile_spec having renamed the saved `window` value.
        configs.set("_window".to_string(), Value::integer(7));
        let mut spec = spec_with_configs(configs);

        let decl = describe(MOVING_AVERAGE_SCRIPT).unwrap();
        apply_declaration(&mut spec, &decl).unwrap();

        assert_eq!(spec.inputs.as_ref().unwrap(), &["value", "reset"]);
        assert_eq!(spec.outputs.as_ref().unwrap(), &["avg"]);

        let configs = spec.configs.as_ref().unwrap();
        assert!(matches!(configs.get("window").unwrap(), Value::Integer(7)));
        assert!(!configs.contains_key("_window"));
        assert_eq!(configs.get_integer_or(CONFIG_TIME_LIMIT_MS, 0), 1234);
        assert_eq!(
            configs.get_string_or_default(CONFIG_SCRIPT),
            MOVING_AVERAGE_SCRIPT
        );

        let config_specs = spec.config_specs.as_ref().unwrap();
        let keys: Vec<&str> = config_specs.keys().map(|k| k.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                CONFIG_SCRIPT,
                "window",
                CONFIG_TIME_LIMIT_MS,
                CONFIG_MEMORY_LIMIT_MB
            ]
        );
        assert_eq!(
            config_specs.get("window").unwrap().title.as_deref(),
            Some("Window")
        );
    }

    #[test]
    fn apply_declaration_prefers_current_value_over_fallback() {
        let mut configs = ModuleConfigs::new();
        configs.set("window".to_string(), Value::integer(3));
        configs.set("_window".to_string(), Value::integer(7));
        let mut spec = spec_with_configs(configs);

        let decl = describe(MOVING_AVERAGE_SCRIPT).unwrap();
        apply_declaration(&mut spec, &decl).unwrap();

        assert!(matches!(
            spec.configs.as_ref().unwrap().get("window").unwrap(),
            Value::Integer(3)
        ));
    }

    #[test]
    fn apply_declaration_uses_declared_default_when_no_value_stored() {
        let mut spec = spec_with_configs(ModuleConfigs::new());
        let decl = describe(MOVING_AVERAGE_SCRIPT).unwrap();
        apply_declaration(&mut spec, &decl).unwrap();
        assert!(matches!(
            spec.configs.as_ref().unwrap().get("window").unwrap(),
            Value::Integer(5)
        ));
    }

    fn process_handler<'a>(
        outputs: &[&str],
        configs: ModuleConfigs,
        state: &'a mut HashMap<String, Value>,
    ) -> ProcessHandler<'a> {
        ProcessHandler {
            ctx: ModuleContext::new(),
            module_id: "test".to_string(),
            configs,
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            state,
            emits: Vec::new(),
        }
    }

    async fn run_process(
        handler: &mut ProcessHandler<'_>,
        script: &str,
        value: Value,
    ) -> Result<()> {
        let source = format!("{script}\nawait onInput({INPUT_PORT_VAR}, {INPUT_VALUE_VAR})");
        run_zapcode(
            source,
            vec![
                (INPUT_PORT_VAR.to_string(), Value::string("value")),
                (INPUT_VALUE_VAR.to_string(), value),
            ],
            EXTERNALS.iter().map(|s| s.to_string()).collect(),
            ResourceLimits::default(),
            handler,
        )
        .await
        .map(|_| ())
    }

    static SUM_SCRIPT: &str = r#"
function onInput(port, value) {
  const prev = getState("total");
  const total = (prev === null ? 0 : prev) + value;
  setState("total", total);
  emit("sum", total);
}
"#;

    #[tokio::test]
    async fn process_collects_emits_and_keeps_state_across_runs() {
        let mut state = HashMap::new();
        let mut handler = process_handler(&["sum"], ModuleConfigs::new(), &mut state);

        run_process(&mut handler, SUM_SCRIPT, Value::integer(2))
            .await
            .unwrap();
        assert!(matches!(
            handler.state.get("total"),
            Some(Value::Integer(2))
        ));

        // Second run sees the state left by the first one.
        run_process(&mut handler, SUM_SCRIPT, Value::integer(3))
            .await
            .unwrap();
        assert_eq!(handler.emits.len(), 2);
        assert_eq!(handler.emits[0].0, "sum");
        assert!(matches!(handler.emits[0].1, Value::Integer(2)));
        assert!(matches!(handler.emits[1].1, Value::Integer(5)));
    }

    #[tokio::test]
    async fn process_reads_config_values() {
        let mut configs = ModuleConfigs::new();
        configs.set("window".to_string(), Value::integer(5));
        let mut state = HashMap::new();
        let mut handler = process_handler(&["value"], configs, &mut state);

        let script = r#"
function onInput(port, value) {
  emit("value", getConfig("window"));
  emit("value", getConfig("missing"));
}
"#;
        run_process(&mut handler, script, Value::Unit)
            .await
            .unwrap();
        assert!(matches!(handler.emits[0].1, Value::Integer(5)));
        assert!(matches!(handler.emits[1].1, Value::Unit));
    }

    #[tokio::test]
    async fn emit_to_undeclared_port_errors() {
        let mut state = HashMap::new();
        let mut handler = process_handler(&["sum"], ModuleConfigs::new(), &mut state);
        let script = r#"
function onInput(port, value) {
  emit("nope", value);
}
"#;
        let err = run_process(&mut handler, script, Value::integer(1))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("undeclared output port"), "{err}");
        assert!(handler.emits.is_empty());
    }
}
