use pyo3::PyTypeInfo;
use pyo3::exceptions::PyRuntimeWarning;
use pyo3::prelude::*;
use std::ffi::CString;
use std::fmt;
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::{
    layer::{Context, Layer},
    registry::LookupSpan,
};

pub struct PythonWarningLayer;

pub fn tracing_init() {
    use tracing_subscriber::prelude::*;

    let layer = PythonWarningLayer;

    let subscriber = tracing_subscriber::registry().with(layer);

    let _ = tracing::subscriber::set_global_default(subscriber);
}

#[derive(Default)]
struct MessageVisitor {
    message: Option<String>,
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            self.message = Some(format!("{:?}", value));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        }
    }
}

impl<S> Layer<S> for PythonWarningLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();

        if metadata.level() > &tracing::Level::WARN {
            return;
        }

        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);

        if let Some(message) = visitor.message {
            Python::with_gil(|py| {
                let message = CString::new(message).expect(
                    "tracing subscriber on_event handler: warning message contained zero byte",
                );

                let _ = PyErr::warn(py, &PyRuntimeWarning::type_object(py), &message, 0);
            });
        }
    }
}
