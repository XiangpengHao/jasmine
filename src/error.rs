#[derive(Debug)]
pub enum JasmineError {
    NeedRetry,
    EvictFailure,
}

impl std::fmt::Display for JasmineError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            JasmineError::NeedRetry => write!(f, "NeedRetry"),
            JasmineError::EvictFailure => write!(f, "EvictFailure"),
        }
    }
}

impl std::error::Error for JasmineError {}
