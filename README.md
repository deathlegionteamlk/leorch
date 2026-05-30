<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:FF4500,100:FF6B00&height=200&section=header&text=Leorch&fontSize=80&fontColor=FFFFFF&animation=fadeIn&fontAlignY=38&desc=A%20PyTorch-like%20Deep%20Learning%20Framework%20in%20Rust&descAlignY=60&descSize=18" width="100%"/>
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=22&pause=1000&color=FF4500&center=true&vCenter=true&width=600&lines=Tensors.+Autograd.+Neural+Nets.+In+Rust.;Built+by+Death+Legion+Team+LK;Fast.+Safe.+Expressive." alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.70%2B-FF4500?style=for-the-badge&logo=rust&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge&logo=opensourceinitiative&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Learning%20Project-orange?style=for-the-badge&logo=bookstack&logoColor=white"/>
  <img src="https://img.shields.io/badge/ndarray-Powered-blueviolet?style=for-the-badge&logo=numpy&logoColor=white"/>
</p>

<p align="center">
  <img src="https://github-profile-trophy.vercel.app/?username=deathlegionteamlk&theme=radical&no-frame=true&row=1&column=4" />
</p>

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=FF4500&height=3&width=100%25"/>
</p>

## <img src="https://media.giphy.com/media/WUlplcMpOCEmTGBtBW/giphy.gif" width="30"> What Is This?

Leorch is a deep learning framework written in Rust, built to replicate the core ideas behind PyTorch — tensors, autograd, layers, optimizers, all of it. It's a ground-up implementation, not a wrapper. If you want to understand how these things actually work, reading and running this codebase is a good way to find out.

Built by **LEGION CODER DEMO AND HEXA** / Death Legion Team LK.

---

## <img src="https://media.giphy.com/media/VgCDAzcKvsR6OM0uWg/giphy.gif" width="30"> Features

<p align="center">
  <img src="https://skillicons.dev/icons?i=rust" height="40"/>
</p>

| Area | What's Included |
|---|---|
| **Tensors** | N-dimensional arrays, math ops, shape ops, indexing |
| **Autograd** | Computational graph, backward pass, gradient tracking |
| **Layers** | Linear, Conv2D, BatchNorm, LayerNorm, Dropout, MaxPool2d, AvgPool2d |
| **Activations** | ReLU, LeakyReLU, Sigmoid, Tanh, GELU, ELU, SELU, Softmax |
| **Loss Functions** | MSE, CrossEntropy, BCE, BCEWithLogits, L1, SmoothL1 |
| **Optimizers** | SGD, Adam, AdamW, RMSprop — all with momentum and weight decay |
| **LR Schedulers** | StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau |
| **Data Loading** | Dataset abstractions, DataLoader with batching and shuffling |
| **Serialization** | Save and load tensors via serde |

---

## <img src="https://media.giphy.com/media/coxQHKASG60HrHtvkt/giphy.gif" width="30"> Installation

```toml
[dependencies]
leorch = { path = "path/to/leorch" }
```

Or clone directly:

```bash
git clone https://github.com/deathlegionteamlk/leorch.git
cd leorch
cargo build --release
```

---

## <img src="https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif" width="28"> Quick Start

```rust
use leorch::tensor::Tensor;
use leorch::nn::{Module, Linear, ReLU, Sequential};
use leorch::loss::{Loss, MSELoss};

fn main() {
    let x = Tensor::randn(&[32, 10]);
    let y = Tensor::randn(&[32, 1]);

    let model = Linear::new(10, 1);
    let output = model.forward(&x);

    let criterion = MSELoss::new();
    let loss = criterion.forward(&output, &y);

    println!("Loss: {}", loss.to_vec()[0]);
}
```

---

## <img src="https://media.giphy.com/media/LMt9638dO8dftAjtco/giphy.gif" width="30"> Examples

### XOR Problem

```bash
cargo run --example xor
```

```rust
use leorch::tensor::Tensor;
use leorch::nn::{Module, Linear, ReLU, Sigmoid};
use leorch::loss::{Loss, MSELoss};

fn main() {
    let x = Tensor::from_slice(&[
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ], &[4, 2]).unwrap();

    let y = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], &[4, 1]).unwrap();

    let hidden = Linear::new(2, 4);
    let activation = ReLU::new();
    let output = Linear::new(4, 1);
    let sigmoid = Sigmoid::new();

    let h = activation.forward(&hidden.forward(&x));
    let pred = sigmoid.forward(&output.forward(&h));

    println!("Predictions: {:?}", pred.to_vec());
}
```

---

## <img src="https://media.giphy.com/media/SWoSkN6DxTszqIKEqv/giphy.gif" width="28"> API Reference

### Tensor Operations

```rust
let zeros = Tensor::zeros(&[2, 3]);
let ones = Tensor::ones(&[2, 3]);
let random = Tensor::randn(&[2, 3]);
let from_data = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();

let reshaped = tensor.reshape(&[3, 2]).unwrap();
let flattened = tensor.flatten();
let transposed = tensor.transpose(0, 1).unwrap();

let sum = tensor.sum();
let mean = tensor.mean();
let max = tensor.max();
let min = tensor.min();

let sqrt = tensor.sqrt();
let exp = tensor.exp();
let log = tensor.log();
let pow = tensor.pow(2.0);

let product = a.matmul(&b).unwrap();
```

### Neural Network Layers

```rust
use leorch::nn::{Linear, Conv2d, BatchNorm2d, Dropout, MaxPool2d, Flatten};

let linear = Linear::new(784, 128);

let conv = Conv2d::new(3, 64, (3, 3));
let conv_with_params = Conv2d::new_with_params(
    3, 64, (3, 3), (1, 1), (1, 1), (1, 1), true,
);

let bn = BatchNorm2d::new(64);
let ln = LayerNorm::new(vec![128]);
let dropout = Dropout::new(0.5);

let maxpool = MaxPool2d::new((2, 2));
let avgpool = AvgPool2d::new((2, 2));

let flatten = Flatten::new();
```

### Activation Functions

```rust
use leorch::nn::{ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, GELU, ELU, SELU};

let relu = ReLU::new();
let leaky_relu = LeakyReLU::with_slope(0.01);
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();
let softmax = Softmax::new(-1);
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

let mse_sum = MSELoss::with_reduction(Reduction::Sum);
```

### Optimizers

```rust
use leorch::optim::{SGD, Adam, AdamW, RMSprop};

let sgd = SGD::new(params, 0.01);
let sgd_momentum = SGD::with_momentum(params, 0.01, 0.9);

let adam = Adam::new(params, 0.001);
let adam_custom = Adam::with_betas(params, 0.001, (0.9, 0.999));

let adamw = AdamW::new(params, 0.001);
let rmsprop = RMSprop::new(params, 0.01);
```

### Learning Rate Schedulers

```rust
use leorch::optim::{StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR};

let step_lr = StepLR::new(0.1, 10, 0.1);
let multi_step = MultiStepLR::new(0.1, vec![30, 60, 90], 0.1);
let exp_lr = ExponentialLR::new(0.1, 0.95);
let cosine = CosineAnnealingLR::new(0.1, 100);
```

### Data Loading

```rust
use leorch::data::{TensorDataset, DataLoader, xor_dataset};

let data = Tensor::randn(&[100, 10]);
let targets = Tensor::randn(&[100, 1]);
let dataset = TensorDataset::new(data, targets).unwrap();

let mut loader = DataLoader::new(dataset, 32);
loader.shuffle();

for (batch_data, batch_targets) in &mut loader {
    // training loop
}

let xor = xor_dataset();
```

---

## <img src="https://media.giphy.com/media/KzJkzjggfGN5Py6nkT/giphy.gif" width="28"> Project Structure

```
leorch/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── tensor.rs
│   ├── autograd.rs
│   ├── nn/
│   │   ├── mod.rs
│   │   ├── layers.rs
│   │   └── activations.rs
│   ├── optim.rs
│   ├── loss.rs
│   ├── functional.rs
│   ├── data.rs
│   └── error.rs
├── examples/
│   └── xor.rs
└── tests/
    └── integration_tests.rs
```

---

## <img src="https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif" width="28"> Testing

```bash
cargo test

cargo test -- --nocapture

cargo test test_tensor_creation
```

---

## <img src="https://media.giphy.com/media/ln7z2eWriiQAllfVcn/giphy.gif" width="28"> Building

```bash
cargo build

cargo build --release

cargo run --example xor
```

---

## <img src="https://media.giphy.com/media/juua9i2c2fA0AIp2iq/giphy.gif" width="28"> Architecture

Leorch is split into five self-contained layers that talk to each other cleanly:

1. **Tensor Core** — built on `ndarray` for efficient n-dimensional operations
2. **Autograd** — a computational graph that tracks operations and runs backward passes
3. **NN Module** — a trait-based layer system; add your own layers without touching core
4. **Optimizers** — stateful, with momentum and adaptive learning rates
5. **Data** — dataset abstractions and a batch-aware data loader

---

## <img src="https://media.giphy.com/media/PjJ1cLHqLEveXysGDB/giphy.gif" width="28"> Performance

CPU operations are parallelized via `rayon`. Memory layout follows `ndarray` conventions, so batch processing is efficient by default. No GPU yet — that's on the roadmap.

---

## <img src="https://media.giphy.com/media/dxn6fRlTIShoeBr69N/giphy.gif" width="28"> Roadmap

- [ ] GPU support via CUDA bindings
- [ ] LSTM, GRU, Transformer layers
- [ ] Full model serialization (save/load entire models)
- [ ] Mixed precision training
- [ ] Distributed training support
- [ ] More loss functions and metrics

---

## <img src="https://media.giphy.com/media/LnQjpWaON8nhr21vNW/giphy.gif" width="28"> Contributing

Pull requests are welcome. If you find a bug or want to add something, open an issue first so we can talk through it.

---

## <img src="https://media.giphy.com/media/VgCDAzcKvsR6OM0uWg/giphy.gif" width="24"> Acknowledgments

Built on `ndarray` for tensor ops, `rayon` for parallelism, and `serde` for serialization. Inspired by PyTorch's API and design philosophy.

---

## <img src="https://media.giphy.com/media/W5eoZHPpUx9sapR0eu/giphy.gif" width="24"> License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:FF4500,100:FF6B00&height=120&section=footer&animation=fadeIn"/>
</p>

<p align="center">
  <sub>Made with <img src="https://img.shields.io/badge/Rust-FF4500?style=flat&logo=rust&logoColor=white"/> by Death Legion Team LK · <a href="https://github.com/deathlegionteamlk/leorch">GitHub</a></sub>
</p>
