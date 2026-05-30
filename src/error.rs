//! Error types for Leorch

use thiserror::Error;

/// Result type alias for Leorch operations
pub type Result<T> = std::result::Result<T, LeorchError>;

/// Error types for Leorch operations
#[derive(Error, Debug)]
pub enum LeorchError {
    /// Shape mismatch error
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    /// Dimension error
    #[error("Dimension error: {0}")]
    DimensionError(String),

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Index out of bounds
    #[error("Index out of bounds: index {index} is out of bounds for dimension {dim} with size {size}")]
    IndexOutOfBounds { index: usize, dim: usize, size: usize },

    /// Gradient error
    #[error("Gradient error: {0}")]
    GradientError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Ndarray error
    #[error("Ndarray error: {0}")]
    NdarrayError(String),
}

impl From<ndarray::ShapeError> for LeorchError {
    fn from(err: ndarray::ShapeError) -> Self {
        LeorchError::NdarrayError(err.to_string())
    }
}
