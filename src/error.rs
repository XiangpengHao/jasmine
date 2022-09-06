#[derive(Debug)]
pub enum JasmineError {
    NeedRetry,
    CacheEmpty,
    ProbeLimitExceeded,
    EvictFailure,
}

impl std::fmt::Display for JasmineError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            JasmineError::NeedRetry => write!(f, "NeedRetry"),
            JasmineError::EvictFailure => write!(f, "EvictFailure"),
            JasmineError::ProbeLimitExceeded => write!(f, "ProbeLimitExceeded"),
            JasmineError::CacheEmpty => write!(f, "CacheEmpty"),
        }
    }
}

impl std::error::Error for JasmineError {}
