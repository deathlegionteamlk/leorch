use crate::tensor::{Tensor, TensorDtype};
use crate::nn::Module;

#[derive(Debug, Clone)]
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        Tensor {
            data: input.data().mapv(|x| x.max(0.0)),
            device: input.device,
            requires_grad: input.requires_grad,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LeakyReLU {
    negative_slope: TensorDtype,
}

impl LeakyReLU {
    pub fn new() -> Self {
        Self::with_slope(0.01)
    }
    
    pub fn with_slope(negative_slope: TensorDtype) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        Tensor {
            data: input.data().mapv(|x| if x > 0.0 { x } else { self.negative_slope * x }),
            device: input.device,
            requires_grad: input.requires_grad,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        Tensor {
            data: input.data().mapv(|x| 1.0 / (1.0 + (-x).exp())),
            device: input.device,
            requires_grad: input.requires_grad,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.tanh()
    }
}

#[derive(Debug, Clone)]
pub struct Softmax {
    dim: Option<usize>,
}

impl Softmax {
    pub fn new() -> Self {
        Self { dim: None }
    }
    
    pub fn with_dim(dim: usize) -> Self {
        Self { dim: Some(dim) }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dim = self.dim.unwrap_or(input.ndim() - 1);
        let max_val = input.max();
        let shifted = input.add_scalar(-max_val);
        let exp = shifted.exp();
        let sum = exp.sum_dim(dim, true).unwrap();
        exp.div(&sum).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct LogSoftmax {
    dim: Option<usize>,
}

impl LogSoftmax {
    pub fn new() -> Self {
        Self { dim: None }
    }
    
    pub fn with_dim(dim: usize) -> Self {
        Self { dim: Some(dim) }
    }
}

impl Default for LogSoftmax {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for LogSoftmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dim = self.dim.unwrap_or(input.ndim() - 1);
        let max_val = input.max();
        let shifted = input.add_scalar(-max_val);
        let exp = shifted.exp();
        let sum = exp.sum_dim(dim, true).unwrap();
        let log_sum = sum.log();
        shifted.sub(&log_sum).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct GELU;

impl GELU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        const SQRT_2_OVER_PI: TensorDtype = 0.7978845608;
        const COEFF: TensorDtype = 0.044715;

        let x_cubed = input.pow(3.0);
        let inner = input.add(&x_cubed.mul_scalar(COEFF)).unwrap();
        let tanh_arg = inner.mul_scalar(SQRT_2_OVER_PI);
        let tanh_result = tanh_arg.tanh();
        let half = tanh_result.add_scalar(1.0);
        half.mul_scalar(0.5).mul(input).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct ELU {
    alpha: TensorDtype,
}

impl ELU {
    pub fn new() -> Self {
        Self::with_alpha(1.0)
    }
    
    pub fn with_alpha(alpha: TensorDtype) -> Self {
        Self { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        Tensor {
            data: input.data().mapv(|x| {
                if x > 0.0 {
                    x
                } else {
                    self.alpha * (x.exp() - 1.0)
                }
            }),
            device: input.device,
            requires_grad: input.requires_grad,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SELU;

impl SELU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        const ALPHA: TensorDtype = 1.6732632423543772848170429916717;
        const SCALE: TensorDtype = 1.0507009873554804934193349852948;

        Tensor {
            data: input.data().mapv(|x| {
                SCALE * if x > 0.0 {
                    x
                } else {
                    ALPHA * (x.exp() - 1.0)
                }
            }),
            device: input.device,
            requires_grad: input.requires_grad,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Swish;

impl Swish {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Swish {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Swish {
    fn forward(&self, input: &Tensor) -> Tensor {
        let sigmoid = input.sigmoid();
        input.mul(&sigmoid).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct HardSigmoid;

impl HardSigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HardSigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for HardSigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        Tensor {
            data: input.data().mapv(|x| {
                if x >= 3.0 {
                    1.0
                } else if x <= -3.0 {
                    0.0
                } else {
                    x / 6.0 + 0.5
                }
            }),
            device: input.device,
            requires_grad: input.requires_grad,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HardSwish;

impl HardSwish {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HardSwish {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for HardSwish {
    fn forward(&self, input: &Tensor) -> Tensor {
        Tensor {
            data: input.data().mapv(|x| {
                let relu6 = x.max(0.0).min(6.0);
                x * relu6 / 6.0
            }),
            device: input.device,
            requires_grad: input.requires_grad,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mish;

impl Mish {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Mish {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Mish {
    fn forward(&self, input: &Tensor) -> Tensor {
        let softplus = input.data().mapv(|x: TensorDtype| (1.0 + x.exp()).ln());
        let softplus_tensor = Tensor::from_array(softplus);
        let tanh = softplus_tensor.tanh();
        input.mul(&tanh).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct Identity;

impl Identity {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Identity {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    }
}
