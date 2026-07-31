use std::sync::Arc;

use indexmap::IndexMap;
use modular_agent_core::{AgentError, AgentValue};
use zapcode_core::Value;

/// Converts an `AgentValue` into a ZapCode `Value` for use as a script input.
///
/// Asymmetries with [`zapcode_to_agent_value`]: `Unit` maps to `Null` (scripts
/// see `null`), while both `null` and `undefined` results map back to `Unit`.
pub(crate) fn agent_value_to_zapcode(value: &AgentValue) -> Value {
    match value {
        AgentValue::Unit => Value::Null,
        AgentValue::Boolean(b) => Value::Bool(*b),
        AgentValue::Integer(i) => Value::Int(*i),
        AgentValue::Number(n) => Value::Float(*n),
        AgentValue::String(s) => Value::String(Arc::from(s.as_str())),
        AgentValue::Array(arr) => Value::Array(arr.iter().map(agent_value_to_zapcode).collect()),
        AgentValue::Object(map) => {
            // im::HashMap iteration order is nondeterministic; sort keys so
            // scripts observe a stable property order across runs.
            let mut entries: Vec<(&String, &AgentValue)> = map.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            let obj: IndexMap<Arc<str>, Value> = entries
                .into_iter()
                .map(|(k, v)| (Arc::<str>::from(k.as_str()), agent_value_to_zapcode(v)))
                .collect();
            Value::Object(obj)
        }
        AgentValue::Tensor(t) => Value::Array(t.iter().map(|f| Value::Float(*f as f64)).collect()),
        // Messages go through their JSON form so scripts receive an object
        // (with `content` staying an array for block content), not a JSON string.
        AgentValue::Message(_) => json_value_to_zapcode(value.to_json()),
        AgentValue::Error(e) => Value::String(Arc::from(format!("{e}").as_str())),
        // No meaningful script representation for image data.
        AgentValue::Image(_) => Value::Null,
    }
}

fn json_value_to_zapcode(value: serde_json::Value) -> Value {
    match value {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(b),
        serde_json::Value::Number(n) => match n.as_i64() {
            Some(i) => Value::Int(i),
            None => Value::Float(n.as_f64().unwrap_or(0.0)),
        },
        serde_json::Value::String(s) => Value::String(Arc::from(s.as_str())),
        serde_json::Value::Array(arr) => {
            Value::Array(arr.into_iter().map(json_value_to_zapcode).collect())
        }
        serde_json::Value::Object(map) => {
            let obj: IndexMap<Arc<str>, Value> = map
                .into_iter()
                .map(|(k, v)| (Arc::<str>::from(k.as_str()), json_value_to_zapcode(v)))
                .collect();
            Value::Object(obj)
        }
    }
}

/// Converts a ZapCode script result into an `AgentValue`.
///
/// Both `undefined` and `null` map to `Unit`. Functions, generators, and
/// built-in methods have no data representation and are rejected anywhere in
/// the result tree.
pub(crate) fn zapcode_to_agent_value(value: Value) -> Result<AgentValue, AgentError> {
    match value {
        Value::Undefined | Value::Null => Ok(AgentValue::Unit),
        Value::Bool(b) => Ok(AgentValue::Boolean(b)),
        Value::Int(i) => Ok(AgentValue::Integer(i)),
        Value::Float(f) => Ok(AgentValue::Number(f)),
        Value::String(s) => Ok(AgentValue::string(s.as_ref())),
        Value::Array(items) => {
            let arr: im::Vector<AgentValue> = items
                .into_iter()
                .map(zapcode_to_agent_value)
                .collect::<Result<_, _>>()?;
            Ok(AgentValue::Array(arr))
        }
        Value::Object(map) => {
            let obj: im::HashMap<String, AgentValue> = map
                .into_iter()
                .map(|(k, v)| Ok((k.to_string(), zapcode_to_agent_value(v)?)))
                .collect::<Result<_, AgentError>>()?;
            Ok(AgentValue::Object(obj))
        }
        Value::Function(_) | Value::Generator(_) | Value::BuiltinMethod { .. } => {
            let kind = match value {
                Value::Generator(_) => "generator",
                _ => "function",
            };
            Err(AgentError::InvalidValue(format!(
                "ZapCode result contains a {kind}: the value of the last expression in the \
                 script is the output — did you forget to call the function \
                 (e.g. `myFunc()` instead of `myFunc`)?"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modular_agent_core::{ContentBlock, Message, MessageContent};
    use zapcode_core::value::{Closure, FunctionId};

    fn dummy_function() -> Value {
        Value::Function(Closure {
            func_id: FunctionId(0),
            captured: vec![],
        })
    }

    #[test]
    fn integer_and_number_convert_both_ways() {
        assert!(matches!(
            agent_value_to_zapcode(&AgentValue::Integer(42)),
            Value::Int(42)
        ));
        assert!(matches!(
            agent_value_to_zapcode(&AgentValue::Number(2.5)),
            Value::Float(f) if (f - 2.5).abs() < 1e-9
        ));
        assert!(matches!(
            zapcode_to_agent_value(Value::Int(42)),
            Ok(AgentValue::Integer(42))
        ));
        assert!(matches!(
            zapcode_to_agent_value(Value::Float(2.5)),
            Ok(AgentValue::Number(f)) if (f - 2.5).abs() < 1e-9
        ));
    }

    #[test]
    fn undefined_and_null_become_unit() {
        assert!(matches!(
            zapcode_to_agent_value(Value::Undefined),
            Ok(AgentValue::Unit)
        ));
        assert!(matches!(
            zapcode_to_agent_value(Value::Null),
            Ok(AgentValue::Unit)
        ));
    }

    #[test]
    fn object_keys_are_sorted() {
        let mut map = im::HashMap::new();
        map.insert("zebra".to_string(), AgentValue::Integer(1));
        map.insert("apple".to_string(), AgentValue::Integer(2));
        map.insert("mango".to_string(), AgentValue::Integer(3));
        let Value::Object(obj) = agent_value_to_zapcode(&AgentValue::Object(map)) else {
            panic!("expected Object to convert to an Object");
        };
        let keys: Vec<&str> = obj.keys().map(|k| k.as_ref()).collect();
        assert_eq!(keys, vec!["apple", "mango", "zebra"]);
    }

    #[test]
    fn message_input_becomes_object() {
        let value = AgentValue::message(Message::user("hi".into()));
        let Value::Object(obj) = agent_value_to_zapcode(&value) else {
            panic!("expected Message to convert to an Object");
        };
        assert!(matches!(obj.get("role"), Some(Value::String(s)) if s.as_ref() == "user"));
        assert!(matches!(obj.get("content"), Some(Value::String(s)) if s.as_ref() == "hi"));
    }

    #[test]
    fn nested_message_in_object_becomes_object() {
        let mut map = im::HashMap::new();
        map.insert(
            "message".to_string(),
            AgentValue::message(Message::user("hi".into())),
        );
        let Value::Object(obj) = agent_value_to_zapcode(&AgentValue::Object(map)) else {
            panic!("expected Object to convert to an Object");
        };
        let Some(Value::Object(inner)) = obj.get("message") else {
            panic!("expected nested message to be an Object");
        };
        assert!(matches!(inner.get("role"), Some(Value::String(s)) if s.as_ref() == "user"));
        assert!(matches!(inner.get("content"), Some(Value::String(s)) if s.as_ref() == "hi"));
    }

    #[test]
    fn message_with_block_content_becomes_array_of_objects() {
        // A message carrying a non-text block (thinking) serializes `content`
        // as a tagged array, not a string, so scripts see `value.content` as
        // an array of block objects.
        let msg = Message {
            role: "assistant".to_string(),
            content: MessageContent::Blocks(vec![
                ContentBlock::Thinking {
                    thinking: "reasoning".to_string(),
                    signature: None,
                    redacted: false,
                },
                ContentBlock::Text {
                    text: "hi".to_string(),
                },
            ]),
            ..Default::default()
        };
        let Value::Object(obj) = agent_value_to_zapcode(&AgentValue::message(msg)) else {
            panic!("expected Message to convert to an Object");
        };
        let Some(Value::Array(blocks)) = obj.get("content") else {
            panic!("block content should convert to an Array, not a JSON string");
        };
        assert_eq!(blocks.len(), 2);

        let Value::Object(thinking) = &blocks[0] else {
            panic!("expected block 0 to be an Object");
        };
        assert!(matches!(thinking.get("type"), Some(Value::String(s)) if s.as_ref() == "thinking"));
        assert!(
            matches!(thinking.get("thinking"), Some(Value::String(s)) if s.as_ref() == "reasoning")
        );

        let Value::Object(text) = &blocks[1] else {
            panic!("expected block 1 to be an Object");
        };
        assert!(matches!(text.get("type"), Some(Value::String(s)) if s.as_ref() == "text"));
        assert!(matches!(text.get("text"), Some(Value::String(s)) if s.as_ref() == "hi"));
    }

    #[test]
    fn function_result_is_error() {
        let err = zapcode_to_agent_value(dummy_function()).unwrap_err();
        assert!(err.to_string().contains("did you forget to call"));
    }

    #[test]
    fn nested_function_result_is_error() {
        let err = zapcode_to_agent_value(Value::Array(vec![Value::Int(1), dummy_function()]))
            .unwrap_err();
        assert!(err.to_string().contains("did you forget to call"));

        let mut obj = IndexMap::new();
        obj.insert(Arc::<str>::from("f"), dummy_function());
        let err = zapcode_to_agent_value(Value::Object(obj)).unwrap_err();
        assert!(err.to_string().contains("did you forget to call"));
    }
}
