use modular_agent_core::{AgentError, AgentValue, async_trait};
use tokio::sync::{mpsc, oneshot};
use zapcode_core::{ResourceLimits, Value, VmState, ZapcodeError, ZapcodeRun};

use crate::value::{agent_value_to_zapcode, zapcode_to_agent_value};

/// Handles external function calls made by a running script.
///
/// A suspended VM hands the call name and arguments to the handler; the value
/// it returns resumes the script as the call's result. Returning an error
/// aborts the run.
#[async_trait]
pub(crate) trait ExternalHandler: Send {
    async fn call(&mut self, name: String, args: Vec<AgentValue>)
    -> Result<AgentValue, AgentError>;
}

/// Handler for scripts that declare no external functions; any call is a bug
/// in the caller's `externals` list, so it just errors.
// Only tests run scripts without externals today; production callers all have
// real handlers.
#[cfg(test)]
pub(crate) struct NoExternals;

#[cfg(test)]
#[async_trait]
impl ExternalHandler for NoExternals {
    async fn call(
        &mut self,
        name: String,
        _args: Vec<AgentValue>,
    ) -> Result<AgentValue, AgentError> {
        Err(AgentError::InvalidValue(format!(
            "ZapCode runtime error: external function `{name}` is not available here"
        )))
    }
}

#[derive(Debug)]
pub(crate) struct ZapOutcome {
    pub(crate) value: AgentValue,
    /// Console output captured up to completion or the first external call.
    /// Output emitted after a resume is lost (upstream limitation: the
    /// snapshot carries it internally but exposes no accessor).
    pub(crate) console: String,
}

struct HostRequest {
    name: String,
    args: Vec<AgentValue>,
    respond: oneshot::Sender<Result<AgentValue, AgentError>>,
}

/// Runs a ZapCode script, bridging external function calls to `handler`.
///
/// The VM lives entirely inside one `spawn_blocking` closure; only
/// `AgentValue`s cross the channel between the VM thread and the async side,
/// so the snapshot and zapcode values never move across threads. Each
/// suspension sends a `HostRequest` and blocks until the async side replies
/// with the handler's result.
pub(crate) async fn run_zapcode(
    source: String,
    inputs: Vec<(String, AgentValue)>,
    externals: Vec<String>,
    limits: ResourceLimits,
    handler: &mut dyn ExternalHandler,
) -> Result<ZapOutcome, AgentError> {
    let (req_tx, mut req_rx) = mpsc::channel::<HostRequest>(1);

    let join = tokio::task::spawn_blocking(move || -> Result<ZapOutcome, AgentError> {
        let input_names: Vec<String> = inputs.iter().map(|(name, _)| name.clone()).collect();
        let input_values: Vec<(String, Value)> = inputs
            .iter()
            .map(|(name, v)| (name.clone(), agent_value_to_zapcode(v)))
            .collect();

        let runner =
            ZapcodeRun::new(source, input_names, externals, limits).map_err(map_zapcode_error)?;
        let result = runner.run(input_values).map_err(map_zapcode_error)?;
        let console = result.stdout;
        let mut state = result.state;

        loop {
            match state {
                VmState::Complete(v) => {
                    return Ok(ZapOutcome {
                        value: zapcode_to_agent_value(v)?,
                        console,
                    });
                }
                VmState::Suspended {
                    function_name,
                    args,
                    snapshot,
                } => {
                    let args = args
                        .into_iter()
                        .map(zapcode_to_agent_value)
                        .collect::<Result<Vec<_>, _>>()?;
                    let (respond, reply_rx) = oneshot::channel();
                    req_tx
                        .blocking_send(HostRequest {
                            name: function_name,
                            args,
                            respond,
                        })
                        .map_err(|_| bridge_closed())?;
                    let ret = reply_rx.blocking_recv().map_err(|_| bridge_closed())??;
                    state = snapshot
                        .resume(agent_value_to_zapcode(&ret))
                        .map_err(map_zapcode_error)?;
                }
            }
        }
    });

    // The closure's req_tx drops when it returns, ending this loop.
    while let Some(req) = req_rx.recv().await {
        let result = handler.call(req.name, req.args).await;
        // A closed reply channel means the VM task is already gone; its join
        // result below carries the real error.
        let _ = req.respond.send(result);
    }

    join.await
        .map_err(|e| AgentError::IoError(format!("ZapCode task error: {e}")))?
}

// The async side dropping its channel ends mid-run only if run_zapcode's
// future is cancelled; surface it as an I/O-level failure, not a script error.
fn bridge_closed() -> AgentError {
    AgentError::IoError("ZapCode host bridge closed".into())
}

pub(crate) fn map_zapcode_error(e: ZapcodeError) -> AgentError {
    match e {
        ZapcodeError::ParseError(_)
        | ZapcodeError::UnsupportedSyntax { .. }
        | ZapcodeError::CompileError(_) => {
            AgentError::InvalidValue(format!("ZapCode compile error: {e}"))
        }
        other => AgentError::InvalidValue(format!("ZapCode runtime error: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_limits() -> ResourceLimits {
        ResourceLimits::default()
    }

    async fn run_simple(source: &str) -> Result<ZapOutcome, AgentError> {
        run_zapcode(
            source.to_string(),
            vec![],
            vec![],
            default_limits(),
            &mut NoExternals,
        )
        .await
    }

    #[tokio::test]
    async fn addition_returns_last_expression() {
        let outcome = run_simple("1 + 2").await.unwrap();
        assert!(matches!(outcome.value, AgentValue::Integer(3)));
    }

    struct FakeHandler {
        calls: Vec<(String, Vec<AgentValue>)>,
        replies: Vec<AgentValue>,
    }

    #[async_trait]
    impl ExternalHandler for FakeHandler {
        async fn call(
            &mut self,
            name: String,
            args: Vec<AgentValue>,
        ) -> Result<AgentValue, AgentError> {
            self.calls.push((name, args));
            Ok(self.replies.remove(0))
        }
    }

    #[tokio::test]
    async fn suspend_resume_calls_handler_in_order() {
        let mut handler = FakeHandler {
            calls: vec![],
            replies: vec![AgentValue::Integer(10), AgentValue::Integer(100)],
        };
        let outcome = run_zapcode(
            "const a = await getNum(1);\nconst b = await getNum(a + 1);\na + b".to_string(),
            vec![],
            vec!["getNum".to_string()],
            default_limits(),
            &mut handler,
        )
        .await
        .unwrap();

        assert!(matches!(outcome.value, AgentValue::Integer(110)));
        assert_eq!(handler.calls.len(), 2);
        assert_eq!(handler.calls[0].0, "getNum");
        assert!(matches!(handler.calls[0].1[..], [AgentValue::Integer(1)]));
        assert_eq!(handler.calls[1].0, "getNum");
        assert!(matches!(handler.calls[1].1[..], [AgentValue::Integer(11)]));
    }

    struct FailingHandler;

    #[async_trait]
    impl ExternalHandler for FailingHandler {
        async fn call(
            &mut self,
            _name: String,
            _args: Vec<AgentValue>,
        ) -> Result<AgentValue, AgentError> {
            Err(AgentError::InvalidValue("tool exploded".into()))
        }
    }

    #[tokio::test]
    async fn handler_error_aborts_the_run() {
        let err = run_zapcode(
            "await boom(); 42".to_string(),
            vec![],
            vec!["boom".to_string()],
            default_limits(),
            &mut FailingHandler,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("tool exploded"));
    }

    #[tokio::test]
    async fn undeclared_external_function_errors() {
        let err = run_simple("missingFunc()").await.unwrap_err();
        assert!(err.to_string().contains("ZapCode runtime error"));
    }

    #[tokio::test]
    async fn infinite_loop_hits_time_limit() {
        // Every VM push counts as an allocation, so an empty loop would trip
        // max_allocations long before 50ms; lift it so the clock fires first.
        let limits = ResourceLimits {
            time_limit_ms: 50,
            max_allocations: usize::MAX,
            ..ResourceLimits::default()
        };
        let err = run_zapcode(
            "while (true) {}".to_string(),
            vec![],
            vec![],
            limits,
            &mut NoExternals,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("time limit exceeded"), "{err}");
    }

    #[tokio::test]
    async fn stdout_is_captured() {
        let outcome = run_simple("console.log(\"hello\");\n1").await.unwrap();
        assert_eq!(outcome.console, "hello\n");
        assert!(matches!(outcome.value, AgentValue::Integer(1)));
    }

    #[tokio::test]
    async fn inputs_are_bound_as_globals() {
        let outcome = run_zapcode(
            "value * 2".to_string(),
            vec![("value".to_string(), AgentValue::Integer(21))],
            vec![],
            default_limits(),
            &mut NoExternals,
        )
        .await
        .unwrap();
        assert!(matches!(outcome.value, AgentValue::Integer(42)));
    }

    #[tokio::test]
    async fn compile_error_is_labeled() {
        let err = run_simple("const = ;").await.unwrap_err();
        assert!(err.to_string().contains("ZapCode compile error"));
    }

    // Records current reality: the bridge design does not depend on this, but
    // if it ever stops compiling, the channel confinement becomes load-bearing.
    #[test]
    fn snapshot_and_value_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<zapcode_core::ZapcodeSnapshot>();
        assert_send::<zapcode_core::Value>();
    }
}
