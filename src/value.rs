use std::sync::Arc;

use indexmap::IndexMap;
use modular_agent_core::{Error, Result, Value};
use zapcode_core::Value as ZapValue;

/// Converts an `Value` into a ZapCode `ZapValue` for use as a script input.
///
/// Asymmetries with [`zapcode_to_value`]: `Unit` maps to `Null` (scripts
/// see `null`), while both `null` and `undefined` results map back to `Unit`.
pub(crate) fn value_to_zapcode(value: &Value) -> ZapValue {
    match value {
        Value::Unit => ZapValue::Null,
        Value::Boolean(b) => ZapValue::Bool(*b),
        Value::Integer(i) => ZapValue::Int(*i),
        Value::Number(n) => ZapValue::Float(*n),
        Value::String(s) => ZapValue::String(Arc::from(s.as_str())),
        Value::Array(arr) => ZapValue::Array(arr.iter().map(value_to_zapcode).collect()),
        Value::Object(map) => {
            // im::HashMap iteration order is nondeterministic; sort keys so
            // scripts observe a stable property order across runs.
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            let obj: IndexMap<Arc<str>, ZapValue> = entries
                .into_iter()
                .map(|(k, v)| (Arc::<str>::from(k.as_str()), value_to_zapcode(v)))
                .collect();
            ZapValue::Object(obj)
        }
        Value::Tensor(t) => ZapValue::Array(t.iter().map(|f| ZapValue::Float(*f as f64)).collect()),
        // Messages go through their JSON form so scripts receive an object
        // (with `content` staying an array for block content), not a JSON string.
        Value::Message(_) => json_value_to_zapcode(value.to_json()),
        Value::Error(e) => ZapValue::String(Arc::from(format!("{e}").as_str())),
        // No meaningful script representation for image data.
        Value::Image(_) => ZapValue::Null,
    }
}

fn json_value_to_zapcode(value: serde_json::Value) -> ZapValue {
    match value {
        serde_json::Value::Null => ZapValue::Null,
        serde_json::Value::Bool(b) => ZapValue::Bool(b),
        serde_json::Value::Number(n) => match n.as_i64() {
            Some(i) => ZapValue::Int(i),
            None => ZapValue::Float(n.as_f64().unwrap_or(0.0)),
        },
        serde_json::Value::String(s) => ZapValue::String(Arc::from(s.as_str())),
        serde_json::Value::Array(arr) => {
            ZapValue::Array(arr.into_iter().map(json_value_to_zapcode).collect())
        }
        serde_json::Value::Object(map) => {
            let obj: IndexMap<Arc<str>, ZapValue> = map
                .into_iter()
                .map(|(k, v)| (Arc::<str>::from(k.as_str()), json_value_to_zapcode(v)))
                .collect();
            ZapValue::Object(obj)
        }
    }
}

/// Converts a ZapCode script result into an `Value`.
///
/// Both `undefined` and `null` map to `Unit`. Functions, generators, and
/// built-in methods have no data representation and are rejected anywhere in
/// the result tree.
pub(crate) fn zapcode_to_value(value: ZapValue) -> Result<Value> {
    match value {
        ZapValue::Undefined | ZapValue::Null => Ok(Value::Unit),
        ZapValue::Bool(b) => Ok(Value::Boolean(b)),
        ZapValue::Int(i) => Ok(Value::Integer(i)),
        ZapValue::Float(f) => Ok(Value::Number(f)),
        ZapValue::String(s) => Ok(Value::string(s.as_ref())),
        ZapValue::Array(items) => {
            let arr: im::Vector<Value> = items
                .into_iter()
                .map(zapcode_to_value)
                .collect::<Result<_, _>>()?;
            Ok(Value::Array(arr))
        }
        ZapValue::Object(map) => {
            let obj: im::HashMap<String, Value> = map
                .into_iter()
                .map(|(k, v)| Ok((k.to_string(), zapcode_to_value(v)?)))
                .collect::<Result<_>>()?;
            Ok(Value::Object(obj))
        }
        ZapValue::Function(_) | ZapValue::Generator(_) | ZapValue::BuiltinMethod { .. } => {
            let kind = match value {
                ZapValue::Generator(_) => "generator",
                _ => "function",
            };
            Err(Error::InvalidValue(format!(
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

    fn dummy_function() -> ZapValue {
        ZapValue::Function(Closure {
            func_id: FunctionId(0),
            captured: vec![],
        })
    }

    #[test]
    fn integer_and_number_convert_both_ways() {
        assert!(matches!(
            value_to_zapcode(&Value::Integer(42)),
            ZapValue::Int(42)
        ));
        assert!(matches!(
            value_to_zapcode(&Value::Number(2.5)),
            ZapValue::Float(f) if (f - 2.5).abs() < 1e-9
        ));
        assert!(matches!(
            zapcode_to_value(ZapValue::Int(42)),
            Ok(Value::Integer(42))
        ));
        assert!(matches!(
            zapcode_to_value(ZapValue::Float(2.5)),
            Ok(Value::Number(f)) if (f - 2.5).abs() < 1e-9
        ));
    }

    #[test]
    fn undefined_and_null_become_unit() {
        assert!(matches!(
            zapcode_to_value(ZapValue::Undefined),
            Ok(Value::Unit)
        ));
        assert!(matches!(zapcode_to_value(ZapValue::Null), Ok(Value::Unit)));
    }

    #[test]
    fn object_keys_are_sorted() {
        let mut map = im::HashMap::new();
        map.insert("zebra".to_string(), Value::Integer(1));
        map.insert("apple".to_string(), Value::Integer(2));
        map.insert("mango".to_string(), Value::Integer(3));
        let ZapValue::Object(obj) = value_to_zapcode(&Value::Object(map)) else {
            panic!("expected Object to convert to an Object");
        };
        let keys: Vec<&str> = obj.keys().map(|k| k.as_ref()).collect();
        assert_eq!(keys, vec!["apple", "mango", "zebra"]);
    }

    #[test]
    fn message_input_becomes_object() {
        let value = Value::message(Message::user("hi".into()));
        let ZapValue::Object(obj) = value_to_zapcode(&value) else {
            panic!("expected Message to convert to an Object");
        };
        assert!(matches!(obj.get("role"), Some(ZapValue::String(s)) if s.as_ref() == "user"));
        assert!(matches!(obj.get("content"), Some(ZapValue::String(s)) if s.as_ref() == "hi"));
    }

    #[test]
    fn nested_message_in_object_becomes_object() {
        let mut map = im::HashMap::new();
        map.insert(
            "message".to_string(),
            Value::message(Message::user("hi".into())),
        );
        let ZapValue::Object(obj) = value_to_zapcode(&Value::Object(map)) else {
            panic!("expected Object to convert to an Object");
        };
        let Some(ZapValue::Object(inner)) = obj.get("message") else {
            panic!("expected nested message to be an Object");
        };
        assert!(matches!(inner.get("role"), Some(ZapValue::String(s)) if s.as_ref() == "user"));
        assert!(matches!(inner.get("content"), Some(ZapValue::String(s)) if s.as_ref() == "hi"));
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
        let ZapValue::Object(obj) = value_to_zapcode(&Value::message(msg)) else {
            panic!("expected Message to convert to an Object");
        };
        let Some(ZapValue::Array(blocks)) = obj.get("content") else {
            panic!("block content should convert to an Array, not a JSON string");
        };
        assert_eq!(blocks.len(), 2);

        let ZapValue::Object(thinking) = &blocks[0] else {
            panic!("expected block 0 to be an Object");
        };
        assert!(
            matches!(thinking.get("type"), Some(ZapValue::String(s)) if s.as_ref() == "thinking")
        );
        assert!(
            matches!(thinking.get("thinking"), Some(ZapValue::String(s)) if s.as_ref() == "reasoning")
        );

        let ZapValue::Object(text) = &blocks[1] else {
            panic!("expected block 1 to be an Object");
        };
        assert!(matches!(text.get("type"), Some(ZapValue::String(s)) if s.as_ref() == "text"));
        assert!(matches!(text.get("text"), Some(ZapValue::String(s)) if s.as_ref() == "hi"));
    }

    #[test]
    fn function_result_is_error() {
        let err = zapcode_to_value(dummy_function()).unwrap_err();
        assert!(err.to_string().contains("did you forget to call"));
    }

    #[test]
    fn nested_function_result_is_error() {
        let err = zapcode_to_value(ZapValue::Array(vec![ZapValue::Int(1), dummy_function()]))
            .unwrap_err();
        assert!(err.to_string().contains("did you forget to call"));

        let mut obj = IndexMap::new();
        obj.insert(Arc::<str>::from("f"), dummy_function());
        let err = zapcode_to_value(ZapValue::Object(obj)).unwrap_err();
        assert!(err.to_string().contains("did you forget to call"));
    }
}
