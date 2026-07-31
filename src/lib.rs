// im's Vector<ToolCall> blows the default trait-solver recursion limit when
// auto traits are checked for AsAgent impls (same as modular-agent-llm/monty).
#![recursion_limit = "256"]

mod code;
mod custom;
mod runner;
mod script;
mod tool;
mod value;
