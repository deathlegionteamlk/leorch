<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:FF4500,50:FF6B00,100:FFA500&height=240&section=header&text=⚡%20LEORCH&fontSize=90&fontColor=FFFFFF&animation=fadeIn&fontAlignY=38&desc=A%20PyTorch-like%20Deep%20Learning%20Framework%20Written%20in%20Rust&descAlignY=60&descSize=20&descColor=FFD580" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=900&size=26&pause=800&color=FF4500&center=true&vCenter=true&width=700&lines=Tensors.+Autograd.+Neural+Nets.+In+Rust.;No+wrappers.+Pure+implementation.;Built+by+Death+Legion+Team+LK.;Fast.+Safe.+Expressive.;Train+models+without+leaving+Rust." alt="Typing SVG" />

<br/>

<img src="https://media.giphy.com/media/qgQUggAC3Pfv687qPC/giphy.gif" width="380" alt="coding gif"/>

<br/><br/>

<img src="https://img.shields.io/badge/Rust-1.70%2B-FF4500?style=for-the-badge&logo=rust&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-0075FF?style=for-the-badge&logo=opensourceinitiative&logoColor=white"/>
<img src="https://img.shields.io/badge/ndarray-Powered-8A2BE2?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/rayon-Parallel-00C896?style=for-the-badge&logo=lightning&logoColor=white"/>
<img src="https://img.shields.io/badge/serde-Serialization-E8A427?style=for-the-badge&logo=json&logoColor=white"/>
<img src="https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge&logo=githubactions&logoColor=white"/>
<img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-blueviolet?style=for-the-badge&logo=linux&logoColor=white"/>

<br/><br/>

<img src="https://skillicons.dev/icons?i=rust,github,vscode,linux,git&perline=5" height="45"/>

<br/><br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=4" width="100%"/>

</div>

---

## <img src="https://media.giphy.com/media/WUlplcMpOCEmTGBtBW/giphy.gif" width="32"> What Is Leorch?

<img align="right" src="https://media.giphy.com/media/LaVp0AyqR5bGsC5Cbm/giphy.gif" width="200"/>

Leorch is a deep learning framework written in Rust. Not a wrapper around PyTorch. Not bindings to libtorch. An actual ground-up implementation of tensors, autograd, layers, optimizers — the whole stack — in Rust.

If you've ever used PyTorch and wondered what's happening under the hood, this codebase is a good place to look. The architecture is intentionally close to PyTorch's, so concepts map across cleanly.

Built by **LEGION CODER DEMO AND HEXA** — Death Legion Team LK.

<br clear="right"/>

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=4" width="100%"/>
</div>

## <img src="https://media.giphy.com/media/VgCDAzcKvsR6OM0uWg/giphy.gif" width="32"> Features

<div align="center">

<img src="https://media.giphy.com/media/f3iwJFOVOwuy7K6FFw/giphy.gif" width="500"/>

<br/>

<img src="https://skillicons.dev/icons?i=rust" height="50"/>

</div>

<br/>

| <img src="https://img.shields.io/badge/Area-FF4500?style=flat-square"/> | <img src="https://img.shields.io/badge/What's%20Inside-1a1a2e?style=flat-square"/> |
|---|---|
| <img src="https://media.giphy.com/media/1ym5LJ17vp77BL8X5O/giphy.gif" width="18"/> **Tensors** | N-dimensional arrays, math ops, shape ops, indexing |
| <img src="https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif" width="18"/> **Autograd** | Computational graph, backward pass, gradient tracking |
| <img src="https://media.giphy.com/media/3oKIPEqDGUULpEU0aQ/giphy.gif" width="18"/> **Layers** | Linear, Conv2D, BatchNorm, LayerNorm, Dropout, MaxPool2d, AvgPool2d |
| <img src="https://media.giphy.com/media/ln7z2eWriiQAllfVcn/giphy.gif" width="18"/> **Activations** | ReLU, LeakyReLU, Sigmoid, Tanh, GELU, ELU, SELU, Softmax |
| <img src="https://media.giphy.com/media/LMt9638dO8dftAjtco/giphy.gif" width="18"/> **Loss Functions** | MSE, CrossEntropy, BCE, BCEWithLogits, L1, SmoothL1 |
| <img src="https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif" width="18"/> **Optimizers** | SGD, Adam, AdamW, RMSprop — all with momentum + weight decay |
| <img src="https://media.giphy.com/media/dxn6fRlTIShoeBr69N/giphy.gif" width="18"/> **LR Schedulers** | StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau |
| <img src="https://media.giphy.com/media/jTNG3RF6EwbkpD4LZx/giphy.gif" width="18"/> **Data Loading** | Dataset abstractions, DataLoader with batching and shuffling |
| <img src="https://media.giphy.com/media/KzJkzjggfGN5Py6nkT/giphy.gif" width="18"/> **Serialization** | Save and load tensors via serde |

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=4" width="100%"/>
</div>

## <img src="https://media.giphy.com/media/coxQHKASG60HrHtvkt/giphy.gif" width="32"> Installation

<div align="center">
<img src="https://media.giphy.com/media/du3J3cXyzhj75IOgvA/giphy.gif" width="400"/>
</div>

<br/>

Add to your `Cargo.toml`:

```toml
[dependencies]
leorch = { path = "path/to/leorch" }
```

Or clone and build:

```bash
git clone https://github.com/deathlegionteamlk/leorch.git
cd leorch
cargo build --release
```

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=4" width="100%"/>
</div>

## <img src="https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif" width="30"> Quick Start

<div align="center">
<img src="https://media.giphy.com/media/26tn33aiTi1jkl6H6/giphy.gif" width="420"/>
</div>

<br/>

> **30-second demo** — create tensors, run a linear layer, compute loss.

```rust
use leorch::tensor::Tensor;
use leorch::nn::{Module, Linear};
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

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=4" width="100%"/>
</div>

## <img src="https://media.giphy.com/media/LMt9638dO8dftAjtco/giphy.gif" width="32"> Examples

<div align="center">

<img src="https://media.giphy.com/media/SWoSkN6DxTszqIKEqv/giphy.gif" width="480"/>

<br/>

<img src="https://img.shields.io/badge/▶%20Run%20Locally-cargo%20run%20--example%20xor-FF4500?style=for-the-badge&logo=rust&logoColor=white"/>

</div>

<br/>

---

### <img src="https://media.giphy.com/media/3oKIPEqDGUULpEU0aQ/giphy.gif" width="26"/> Example 1 — XOR Problem

> The classic test for a neural net that can't be solved by a straight line. Two inputs, one output, four data points.

<details>
<summary><b>🔍 Click to expand full source</b></summary>

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

    let hidden    = Linear::new(2, 4);
    let activation = ReLU::new();
    let output    = Linear::new(4, 1);
    let sigmoid   = Sigmoid::new();

    let h    = activation.forward(&hidden.forward(&x));
    let pred = sigmoid.forward(&output.forward(&h));

    println!("Predictions: {:?}", pred.to_vec());
}
```

</details>

**What the network looks like:**

```
Input (2)  ──►  Hidden ReLU (4)  ──►  Output Sigmoid (1)
  [0,0]                                     [0]
  [0,1]    ──►       2→4→1        ──►       [1]
  [1,0]                                     [1]
  [1,1]                                     [0]
```

```bash
cargo run --example xor
```

<div align="center">
<img src="https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif" width="300"/>
</div>

---

### <img src="https://media.giphy.com/media/PjJ1cLHqLEveXysGDB/giphy.gif" width="26"/> Example 2 — Linear Regression

> Simplest possible supervised learning. Fit a line to noisy data using MSE loss and SGD.

<details>
<summary><b>🔍 Click to expand full source</b></summary>

```rust
use leorch::tensor::Tensor;
use leorch::nn::{Module, Linear};
use leorch::loss::{Loss, MSELoss};
use leorch::optim::SGD;

fn main() {
    let x = Tensor::randn(&[64, 1]);
    let y = Tensor::randn(&[64, 1]);

    let model     = Linear::new(1, 1);
    let criterion = MSELoss::new();
    let mut optim = SGD::new(model.parameters(), 0.01);

    for epoch in 0..100 {
        let pred = model.forward(&x);
        let loss = criterion.forward(&pred, &y);

        optim.zero_grad();
        loss.backward();
        optim.step();

        if epoch % 10 == 0 {
            println!("Epoch {epoch}: loss = {:.4}", loss.to_vec()[0]);
        }
    }
}
```

</details>

```bash
cargo run --example linear_regression
```

---

### <img src="https://media.giphy.com/media/dxn6fRlTIShoeBr69N/giphy.gif" width="26"/> Example 3 — Multi-Layer Classifier

> Stack a few layers, throw in BatchNorm and Dropout, train on synthetic class data.

<details>
<summary><b>🔍 Click to expand full source</b></summary>

```rust
use leorch::tensor::Tensor;
use leorch::nn::{Module, Linear, ReLU, BatchNorm1d, Dropout};
use leorch::loss::{Loss, CrossEntropyLoss};
use leorch::optim::Adam;

fn main() {
    let x = Tensor::randn(&[128, 20]);
    let y = Tensor::randint(0, 5, &[128]);

    let fc1      = Linear::new(20, 64);
    let bn1      = BatchNorm1d::new(64);
    let relu     = ReLU::new();
    let dropout  = Dropout::new(0.3);
    let fc2      = Linear::new(64, 5);
    let criterion = CrossEntropyLoss::new();

    let mut params = vec![];
    params.extend(fc1.parameters());
    params.extend(fc2.parameters());

    let mut optim = Adam::new(params, 0.001);

    for epoch in 0..50 {
        let h    = dropout.forward(&relu.forward(&bn1.forward(&fc1.forward(&x))));
        let logits = fc2.forward(&h);
        let loss = criterion.forward(&logits, &y);

        optim.zero_grad();
        loss.backward();
        optim.step();

        if epoch % 10 == 0 {
            println!("Epoch {epoch}: loss = {:.4}", loss.to_vec()[0]);
        }
    }
}
```

</details>

```bash
cargo run --example classifier
```

<div align="center">
<img src="https://media.giphy.com/media/juua9i2c2fA0AIp2iq/giphy.gif" width="320"/>
</div>

---

### <img src="https://media.giphy.com/media/jTNG3RF6EwbkpD4LZx/giphy.gif" width="26"/> Example 4 — Conv2D Image Feature Extraction

> A small conv stack on a random 3-channel image tensor. Tests Conv2d → BatchNorm2d → ReLU → MaxPool2d.

<details>
<summary><b>🔍 Click to expand full source</b></summary>

```rust
use leorch::tensor::Tensor;
use leorch::nn::{Module, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear};
use leorch::loss::{Loss, CrossEntropyLoss};

fn main() {
    let x = Tensor::randn(&[8, 3, 32, 32]);
    let y = Tensor::randint(0, 10, &[8]);

    let conv1   = Conv2d::new(3, 16, (3, 3));
    let bn1     = BatchNorm2d::new(16);
    let relu    = ReLU::new();
    let pool    = MaxPool2d::new((2, 2));
    let flatten = Flatten::new();
    let fc      = Linear::new(16 * 15 * 15, 10);
    let criterion = CrossEntropyLoss::new();

    let h      = pool.forward(&relu.forward(&bn1.forward(&conv1.forward(&x))));
    let flat   = flatten.forward(&h);
    let logits = fc.forward(&flat);
    let loss   = criterion.forward(&logits, &y);

    println!("Conv stack loss: {:.4}", loss.to_vec()[0]);
}
```

</details>

```bash
cargo run --example conv_features
```

---

### <img src="https://media.giphy.com/media/1ym5LJ17vp77BL8X5O/giphy.gif" width="26"/> Example 5 — DataLoader Training Loop

> Load batches from a TensorDataset, shuffle each epoch, full gradient loop.

<details>
<summary><b>🔍 Click to expand full source</b></summary>

```rust
use leorch::tensor::Tensor;
use leorch::nn::{Module, Linear};
use leorch::loss::{Loss, MSELoss};
use leorch::optim::AdamW;
use leorch::data::{TensorDataset, DataLoader};

fn main() {
    let data    = Tensor::randn(&[200, 16]);
    let targets = Tensor::randn(&[200, 1]);
    let dataset = TensorDataset::new(data, targets).unwrap();

    let mut loader = DataLoader::new(dataset, 32);
    loader.shuffle();

    let model     = Linear::new(16, 1);
    let criterion = MSELoss::new();
    let mut optim = AdamW::new(model.parameters(), 0.001);

    for epoch in 0..10 {
        let mut total_loss = 0.0_f32;
        for (batch_x, batch_y) in &mut loader {
            let pred = model.forward(&batch_x);
            let loss = criterion.forward(&pred, &batch_y);

            optim.zero_grad();
            loss.backward();
            optim.step();

            total_loss += loss.to_vec()[0];
        }
        println!("Epoch {epoch}: avg loss = {:.4}", total_loss / 200.0);
    }
}
```

</details>

```bash
cargo run --example dataloader_loop
```

<div align="center">
<img src="https://media.giphy.com/media/f3iwJFOVOwuy7K6FFw/giphy.gif" width="350"/>
</div>

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=4" width="100%"/>
</div>

## <img src="https://media.giphy.com/media/SWoSkN6DxTszqIKEqv/giphy.gif" width="32"> API Reference

<div align="center">

<img src="https://media.giphy.com/media/LaVp0AyqR5bGsC5Cbm/giphy.gif" width="400"/>

<br/>

<img src="https://skillicons.dev/icons?i=rust,github,vscode" height="40"/>

</div>

<br/>

### <img src="https://media.giphy.com/media/1ym5LJ17vp77BL8X5O/giphy.gif" width="24"/> Tensor Operations

<details>
<summary><b>📦 Creation, shape ops, math — click to expand</b></summary>

```rust
let zeros     = Tensor::zeros(&[2, 3]);
let ones      = Tensor::ones(&[2, 3]);
let random    = Tensor::randn(&[2, 3]);
let from_data = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();

let reshaped   = tensor.reshape(&[3, 2]).unwrap();
let flattened  = tensor.flatten();
let transposed = tensor.transpose(0, 1).unwrap();

let sum  = tensor.sum();
let mean = tensor.mean();
let max  = tensor.max();
let min  = tensor.min();

let sqrt = tensor.sqrt();
let exp  = tensor.exp();
let log  = tensor.log();
let pow  = tensor.pow(2.0);

let product = a.matmul(&b).unwrap();
```

</details>

---

### <img src="https://media.giphy.com/media/3oKIPEqDGUULpEU0aQ/giphy.gif" width="24"/> Neural Network Layers

<details>
<summary><b>🧱 Linear, Conv2d, BN, Dropout, Pooling — click to expand</b></summary>

```rust
use leorch::nn::{Linear, Conv2d, BatchNorm2d, Dropout, MaxPool2d, Flatten};

let linear = Linear::new(784, 128);

let conv = Conv2d::new(3, 64, (3, 3));
let conv_with_params = Conv2d::new_with_params(
    3, 64, (3, 3), (1, 1), (1, 1), (1, 1), true,
);

let bn      = BatchNorm2d::new(64);
let ln      = LayerNorm::new(vec![128]);
let dropout = Dropout::new(0.5);

let maxpool = MaxPool2d::new((2, 2));
let avgpool = AvgPool2d::new((2, 2));
let flatten = Flatten::new();
```

</details>

---

### <img src="https://media.giphy.com/media/ln7z2eWriiQAllfVcn/giphy.gif" width="24"/> Activation Functions

<details>
<summary><b>⚡ ReLU, GELU, Sigmoid, Tanh and more — click to expand</b></summary>

```rust
use leorch::nn::{ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, GELU, ELU, SELU};

let relu       = ReLU::new();
let leaky_relu = LeakyReLU::with_slope(0.01);
let sigmoid    = Sigmoid::new();
let tanh       = Tanh::new();
let softmax    = Softmax::new(-1);
let gelu       = GELU::new();
let elu        = ELU::with_alpha(1.0);
let selu       = SELU::new();
```

</details>

---

### <img src="https://media.giphy.com/media/LMt9638dO8dftAjtco/giphy.gif" width="24"/> Loss Functions

<details>
<summary><b>📉 MSE, CrossEntropy, BCE and more — click to expand</b></summary>

```rust
use leorch::loss::{MSELoss, L1Loss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss};

let mse        = MSELoss::new();
let l1         = L1Loss::new();
let ce         = CrossEntropyLoss::new();
let bce        = BCELoss::new();
let bce_logits = BCEWithLogitsLoss::new();

let mse_sum = MSELoss::with_reduction(Reduction::Sum);
```

</details>

---

### <img src="https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif" width="24"/> Optimizers

<details>
<summary><b>🚀 SGD, Adam, AdamW, RMSprop — click to expand</b></summary>

```rust
use leorch::optim::{SGD, Adam, AdamW, RMSprop};

let sgd          = SGD::new(params, 0.01);
let sgd_momentum = SGD::with_momentum(params, 0.01, 0.9);

let adam        = Adam::new(params, 0.001);
let adam_custom = Adam::with_betas(params, 0.001, (0.9, 0.999));

let adamw   = AdamW::new(params, 0.001);
let rmsprop = RMSprop::new(params, 0.01);
```

</details>

---

### <img src="https://media.giphy.com/media/dxn6fRlTIShoeBr69N/giphy.gif" width="24"/> Learning Rate Schedulers

<details>
<summary><b>📊 StepLR, Cosine, Exponential — click to expand</b></summary>

```rust
use leorch::optim::{StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR};

let step_lr    = StepLR::new(0.1, 10, 0.1);
let multi_step = MultiStepLR::new(0.1, vec![30, 60, 90], 0.1);
let exp_lr     = ExponentialLR::new(0.1, 0.95);
let cosine     = CosineAnnealingLR::new(0.1, 100);
```

</details>

---

### <img src="https://media.giphy.com/media/jTNG3RF6EwbkpD4LZx/giphy.gif" width="24"/> Data Loading

<details>
<summary><b>🗂️ TensorDataset, DataLoader, xor_dataset — click to expand</b></summary>

```rust
use leorch::data::{TensorDataset, DataLoader, xor_dataset};

let data    = Tensor::randn(&[100, 10]);
let targets = Tensor::randn(&[100, 1]);
let dataset = TensorDataset::new(data, targets).unwrap();

let mut loader = DataLoader::new(dataset, 32);
loader.shuffle();

for (batch_data, batch_targets) in &mut loader {
    // training loop
}

let xor = xor_dataset();
```

</details>

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=4" width="100%"/>
</div>

## <img src="https://media.giphy.com/media/KzJkzjggfGN5Py6nkT/giphy.gif" width="32"> Project Structure

<div align="center">
<img src="https://media.giphy.com/media/du3J3cXyzhj75IOgvA/giphy.gif" width="380"/>
</div>

<br/>

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
│   ├── xor.rs
│   ├── linear_regression.rs
│   ├── classifier.rs
│   ├── conv_features.rs
│   └── dataloader_loop.rs
└── tests/
    └── integration_tests.rs
```

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=4" width="100%"/>
</div>

## <img src="https://media.giphy.com/media/juua9i2c2fA0AIp2iq/giphy.gif" width="32"> Architecture

<div align="center">
<img src="https://media.giphy.com/media/f3iwJFOVOwuy7K6FFw/giphy.gif" width="380"/>
</div>

<br/>

Five layers, each self-contained, each doing exactly one thing:

| Layer | Built on | What it does |
|---|---|---|
| <img src="https://media.giphy.com/media/1ym5LJ17vp77BL8X5O/giphy.gif" width="16"/> **Tensor Core** | `ndarray` | N-dimensional arrays, all math ops |
| <img src="https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif" width="16"/> **Autograd** | Custom graph | Tracks ops, runs backward pass |
| <img src="https://media.giphy.com/media/3oKIPEqDGUULpEU0aQ/giphy.gif" width="16"/> **NN Module** | Trait system | Extensible layer API |
| <img src="https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif" width="16"/> **Optimizers** | Stateful structs | Momentum, adaptive rates |
| <img src="https://media.giphy.com/media/jTNG3RF6EwbkpD4LZx/giphy.gif" width="16"/> **Data** | Iterator protocol | Batching, shuffling, datasets |

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=4" width="100%"/>
</div>

## <img src="https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif" width="32"> Testing

<div align="center">
<img src="https://media.giphy.com/media/3oKIPEqDGUULpEU0aQ/giphy.gif" width="340"/>
</div>

<br/>

```bash
cargo test

cargo test -- --nocapture

cargo test test_tensor_creation
```

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=4" width="100%"/>
</div>

## <img src="https://media.giphy.com/media/ln7z2eWriiQAllfVcn/giphy.gif" width="32"> Building

```bash
cargo build

cargo build --release

cargo run --example xor
```

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=4" width="100%"/>
</div>

## <img src="https://media.giphy.com/media/PjJ1cLHqLEveXysGDB/giphy.gif" width="32"> Performance

<div align="center">
<img src="https://media.giphy.com/media/26tn33aiTi1jkl6H6/giphy.gif" width="380"/>
</div>

<br/>

CPU ops run in parallel via `rayon`. Memory layout matches `ndarray` conventions, so batch processing doesn't fight the allocator. GPU isn't there yet — CUDA bindings are on the roadmap.

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=4" width="100%"/>
</div>

## <img src="https://media.giphy.com/media/dxn6fRlTIShoeBr69N/giphy.gif" width="32"> Roadmap

<div align="center">

<img src="https://media.giphy.com/media/l0HlBO7eyXzSZkJri/giphy.gif" width="360"/>

</div>

<br/>

- [ ] <img src="https://img.shields.io/badge/GPU-CUDA%20Bindings-76b900?style=flat-square&logo=nvidia&logoColor=white"/> GPU support via CUDA
- [ ] <img src="https://img.shields.io/badge/Layers-LSTM%20%7C%20GRU%20%7C%20Transformer-FF4500?style=flat-square&logo=pytorch&logoColor=white"/> Recurrent and attention layers
- [ ] <img src="https://img.shields.io/badge/Serialization-Full%20Model%20Save%2FLoad-0075FF?style=flat-square&logo=files&logoColor=white"/> Save and load entire models
- [ ] <img src="https://img.shields.io/badge/Training-Mixed%20Precision-8A2BE2?style=flat-square&logo=lightning&logoColor=white"/> FP16 / mixed precision
- [ ] <img src="https://img.shields.io/badge/Scale-Distributed%20Training-E8A427?style=flat-square&logo=kubernetes&logoColor=white"/> Distributed training
- [ ] <img src="https://img.shields.io/badge/Metrics-More%20Loss%20Functions-00C896?style=flat-square&logo=chartdotjs&logoColor=white"/> Expanded loss and metric library

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=4" width="100%"/>
</div>

## <img src="https://media.giphy.com/media/LnQjpWaON8nhr21vNW/giphy.gif" width="32"> Contributing

<div align="center">
<img src="https://media.giphy.com/media/fwbzI2kV3Qrlpkh59e/giphy.gif" width="300"/>
</div>

<br/>

Pull requests are welcome. Open an issue first if you're planning something substantial — saves everyone time if we talk through the design before you write the code.

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=4" width="100%"/>
</div>

## <img src="https://media.giphy.com/media/VgCDAzcKvsR6OM0uWg/giphy.gif" width="28"> Acknowledgments

<div align="center">

<img src="https://skillicons.dev/icons?i=rust,github,vscode,linux,git,vim" height="40"/>

</div>

<br/>

Built on `ndarray` for tensor ops, `rayon` for parallelism, and `serde` for serialization. The API design follows PyTorch closely — if you know PyTorch, you'll find your way around here quickly.

---

## <img src="https://media.giphy.com/media/W5eoZHPpUx9sapR0eu/giphy.gif" width="26"> License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:FF4500,50:FF6B00,100:FFA500&height=160&section=footer&animation=fadeIn&text=Death%20Legion%20Team%20LK&fontSize=28&fontColor=FFFFFF&fontAlignY=65"/>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=14&pause=1200&color=FF4500&center=true&vCenter=true&width=500&lines=cargo+build+--release;cargo+test+--+--nocapture;cargo+run+--example+xor" alt="footer typing"/>

<br/>

<img src="https://img.shields.io/badge/Made%20with-Rust%20🦀-FF4500?style=for-the-badge&logo=rust&logoColor=white"/>
<img src="https://img.shields.io/badge/By-Death%20Legion%20Team%20LK-1a1a2e?style=for-the-badge&logo=github&logoColor=white"/>

<br/><br/>

<img src="https://media.giphy.com/media/du3J3cXyzhj75IOgvA/giphy.gif" width="60"/>

</div>

<!-- Badge earned via contribution -->
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)
