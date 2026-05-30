<p align="center">
  <img src="logo.png" alt="Leorch Logo" width="200"/>
</p>

# Leorch: A PyTorch-like Deep Learning Framework in Rust

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**By LEGION CODER DEMO AND HEXA**

Leorch is a deep learning framework written in Rust that aims to replicate PyTorch's core features, including tensors, automatic differentiation, neural network layers, optimizers, and serialization.

## Features

- **Tensor Operations**: Multi-dimensional arrays with support for various operations including mathematical operations, shape manipulation, and indexing
- **Automatic Differentiation**: Computational graph and backward pass for gradient computation
- **Neural Network Layers**: Linear, Conv2D, BatchNorm, LayerNorm, Dropout, and more
- **Activation Functions**: ReLU, Sigmoid, Tanh, GELU, Softmax, and more
- **Loss Functions**: MSE, Cross Entropy, BCE, L1, Smooth L1, and more
- **Optimizers**: SGD, Adam, AdamW, RMSprop with momentum and weight decay support
- **Learning Rate Schedulers**: StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
- **Data Loading**: Dataset abstractions, DataLoader with batching and shuffling
- **Serialization**: Save and load tensors with serde support

## Installation

Add Leorch to your `Cargo.toml`:

```toml
[dependencies]
leorch = { path = "path/to/leorch" }
```

Or clone the repository:

```bash
git clone https://github.com/deathlegionteamlk/leorch.git
cd leorch
cargo build --release
```

## Quick Start

```rust
use leorch::tensor::Tensor;
use leorch::nn::{Module, Linear, ReLU, Sequential};
use leorch::loss::{Loss, MSELoss};

fn main() {
    // Create tensors
    let x = Tensor::randn(&[32, 10]);
    let y = Tensor::randn(&[32, 1]);

    // Define a model
    let model = Linear::new(10, 1);

    // Forward pass
    let output = model.forward(&x);

    // Compute loss
    let criterion = MSELoss::new();
    let loss = criterion.forward(&output, &y);

    println!("Loss: {}", loss.to_vec()[0]);
}
```

## Examples

### XOR Problem

The XOR example demonstrates training a simple neural network to learn the XOR function:

```bash
cargo run --example xor
```

```rust
use leorch::tensor::Tensor;
use leorch::nn::{Module, Linear, ReLU, Sigmoid};
use leorch::loss::{Loss, MSELoss};

fn main() {
    // XOR dataset
    let x = Tensor::from_slice(&[
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ], &[4, 2]).unwrap();

    let y = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], &[4, 1]).unwrap();

    // Create model: 2 -> 4 -> 1
    let hidden = Linear::new(2, 4);
    let activation = ReLU::new();
    let output = Linear::new(4, 1);
    let sigmoid = Sigmoid::new();

    // Forward pass
    let h = activation.forward(&hidden.forward(&x));
    let pred = sigmoid.forward(&output.forward(&h));

    println!("Predictions: {:?}", pred.to_vec());
}
```

## API Reference

### Tensor Operations

```rust
// Creation
let zeros = Tensor::zeros(&[2, 3]);
let ones = Tensor::ones(&[2, 3]);
let random = Tensor::randn(&[2, 3]);
let from_data = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();

// Shape operations
let reshaped = tensor.reshape(&[3, 2]).unwrap();
let flattened = tensor.flatten();
let transposed = tensor.transpose(0, 1).unwrap();

// Mathematical operations
let sum = tensor.sum();
let mean = tensor.mean();
let max = tensor.max();
let min = tensor.min();

// Element-wise operations
let sqrt = tensor.sqrt();
let exp = tensor.exp();
let log = tensor.log();
let pow = tensor.pow(2.0);

// Matrix operations
let product = a.matmul(&b).unwrap();
```

### Neural Network Layers

```rust
use leorch::nn::{Linear, Conv2d, BatchNorm2d, Dropout, MaxPool2d, Flatten};

// Linear layer
let linear = Linear::new(784, 128);

// Convolutional layer
let conv = Conv2d::new(3, 64, (3, 3));
let conv_with_params = Conv2d::new_with_params(
    3,      // in_channels
    64,     // out_channels
    (3, 3), // kernel_size
    (1, 1), // stride
    (1, 1), // padding
    (1, 1), // dilation
    true,   // use_bias
);

// Normalization
let bn = BatchNorm2d::new(64);
let ln = LayerNorm::new(vec![128]);

// Regularization
let dropout = Dropout::new(0.5);

// Pooling
let maxpool = MaxPool2d::new((2, 2));
let avgpool = AvgPool2d::new((2, 2));

// Utility
let flatten = Flatten::new();
```

### Activation Functions

```rust
use leorch::nn::{ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, GELU, ELU, SELU};

let relu = ReLU::new();
let leaky_relu = LeakyReLU::with_slope(0.01);
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();
let softmax = Softmax::new(-1); // dimension
let gelu = GELU::new();
let elu = ELU::with_alpha(1.0);
let selu = SELU::new();
```

### Loss Functions

```rust
use leorch::loss::{MSELoss, L1Loss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss};

let mse = MSELoss::new();
let l1 = L1Loss::new();
let ce = CrossEntropyLoss::new();
let bce = BCELoss::new();
let bce_logits = BCEWithLogitsLoss::new();

// With custom reduction
let mse_sum = MSELoss::with_reduction(Reduction::Sum);
```

### Optimizers

```rust
use leorch::optim::{SGD, Adam, AdamW, RMSprop};

// SGD
let sgd = SGD::new(params, 0.01);
let sgd_momentum = SGD::with_momentum(params, 0.01, 0.9);

// Adam
let adam = Adam::new(params, 0.001);
let adam_custom = Adam::with_betas(params, 0.001, (0.9, 0.999));

// AdamW
let adamw = AdamW::new(params, 0.001);

// RMSprop
let rmsprop = RMSprop::new(params, 0.01);
```

### Learning Rate Schedulers

```rust
use leorch::optim::{StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR};

let step_lr = StepLR::new(0.1, 10, 0.1); // Decay by 0.1 every 10 epochs
let multi_step = MultiStepLR::new(0.1, vec![30, 60, 90], 0.1);
let exp_lr = ExponentialLR::new(0.1, 0.95);
let cosine = CosineAnnealingLR::new(0.1, 100);
```

### Data Loading

```rust
use leorch::data::{TensorDataset, DataLoader, xor_dataset};

// Create dataset
let data = Tensor::randn(&[100, 10]);
let targets = Tensor::randn(&[100, 1]);
let dataset = TensorDataset::new(data, targets).unwrap();

// Create data loader
let mut loader = DataLoader::new(dataset, 32);
loader.shuffle();

for (batch_data, batch_targets) in &mut loader {
    // Training loop
}

// Pre-built datasets
let xor = xor_dataset();
```

## Project Structure

```
leorch/
├── Cargo.toml          # Project configuration
├── src/
│   ├── lib.rs          # Library entry point
│   ├── tensor.rs       # Tensor implementation
│   ├── autograd.rs     # Automatic differentiation
│   ├── nn/
│   │   ├── mod.rs      # Neural network module
│   │   ├── layers.rs   # Layer implementations
│   │   └── activations.rs # Activation functions
│   ├── optim.rs        # Optimizers and schedulers
│   ├── loss.rs         # Loss functions
│   ├── functional.rs   # Low-level operations
│   ├── data.rs         # Data loading utilities
│   └── error.rs        # Error types
├── examples/
│   └── xor.rs          # XOR example
└── tests/
    └── integration_tests.rs # Integration tests
```

## Testing

Run the test suite:

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_tensor_creation
```

## Building

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run examples
cargo run --example xor
```

## Architecture

Leorch follows a modular architecture inspired by PyTorch:

1. **Tensor Core**: Built on `ndarray` for efficient n-dimensional array operations
2. **Autograd**: Computational graph with automatic differentiation
3. **NN Module**: Trait-based layer system for extensibility
4. **Optimizers**: Stateful optimizers with momentum and adaptive learning rates
5. **Data**: Dataset abstractions and efficient data loading

## Performance

- CPU-optimized operations using `ndarray` and `rayon` for parallelization
- Efficient memory layout for tensor operations
- Batch processing support for data loading

## Future Work

- [ ] GPU support via CUDA bindings
- [ ] Additional layer types (LSTM, GRU, Transformer)
- [ ] More loss functions and metrics
- [ ] Model serialization (save/load entire models)
- [ ] Distributed training support
- [ ] Mixed precision training

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by PyTorch's design and API
- Built with `ndarray` for tensor operations
- Developed by Death Legion Team LK / DEMO X HEXa

## Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This is a learning project and educational implementation. For production use, consider established frameworks like PyTorch, TensorFlow, or JAX.
