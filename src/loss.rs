use crate::tensor::{Tensor, TensorDtype};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reduction {
    None,
    Sum,
    Mean,
}

impl Default for Reduction {
    fn default() -> Self {
        Reduction::Mean
    }
}

pub trait Loss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor;
}

#[derive(Debug, Clone)]
pub struct MSELoss {
    reduction: Reduction,
}

impl MSELoss {
    pub fn new() -> Self {
        Self::with_reduction(Reduction::Mean)
    }
    
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for MSELoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let diff = prediction.sub(target).unwrap();
        let squared = diff.pow(2.0);

        match self.reduction {
            Reduction::None => squared,
            Reduction::Sum => {
                let sum = squared.sum();
                Tensor::from_slice(&[sum], &[1]).unwrap()
            }
            Reduction::Mean => {
                let mean = squared.mean();
                Tensor::from_slice(&[mean], &[1]).unwrap()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct L1Loss {
    reduction: Reduction,
}

impl L1Loss {
    pub fn new() -> Self {
        Self::with_reduction(Reduction::Mean)
    }
    
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl Default for L1Loss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for L1Loss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let diff = prediction.sub(target).unwrap();
        let abs_diff = diff.abs();

        match self.reduction {
            Reduction::None => abs_diff,
            Reduction::Sum => {
                let sum = abs_diff.sum();
                Tensor::from_slice(&[sum], &[1]).unwrap()
            }
            Reduction::Mean => {
                let mean = abs_diff.mean();
                Tensor::from_slice(&[mean], &[1]).unwrap()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SmoothL1Loss {
    delta: TensorDtype,
    reduction: Reduction,
}

impl SmoothL1Loss {
    pub fn new() -> Self {
        Self::with_params(1.0, Reduction::Mean)
    }
    
    pub fn with_params(delta: TensorDtype, reduction: Reduction) -> Self {
        Self { delta, reduction }
    }
}

impl Default for SmoothL1Loss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for SmoothL1Loss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let diff = prediction.sub(target).unwrap();
        let abs_diff = diff.abs();

        let loss_data: Vec<TensorDtype> = diff.data()
            .iter()
            .zip(abs_diff.data().iter())
            .map(|(&d, &ad)| {
                if ad < self.delta {
                    0.5 * d * d
                } else {
                    self.delta * ad - 0.5 * self.delta * self.delta
                }
            })
            .collect();

        let loss = Tensor::from_slice(&loss_data, &diff.shape()).unwrap();

        match self.reduction {
            Reduction::None => loss,
            Reduction::Sum => {
                let sum = loss.sum();
                Tensor::from_slice(&[sum], &[1]).unwrap()
            }
            Reduction::Mean => {
                let mean = loss.mean();
                Tensor::from_slice(&[mean], &[1]).unwrap()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BCELoss {
    reduction: Reduction,
    eps: TensorDtype,
}

impl BCELoss {
    pub fn new() -> Self {
        Self::with_reduction(Reduction::Mean)
    }
    
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self {
            reduction,
            eps: 1e-8,
        }
    }
}

impl Default for BCELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for BCELoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let clamped = prediction.data().mapv(|x| {
            x.max(self.eps).min(1.0 - self.eps)
        });
        let clamped_tensor = Tensor::from_array(clamped);

        let ones = Tensor::ones(&target.shape());
        let one_minus_p = ones.sub(&clamped_tensor).unwrap();
        let one_minus_y = ones.sub(target).unwrap();

        let log_p = clamped_tensor.log();
        let log_one_minus_p = one_minus_p.log();

        let term1 = target.mul(&log_p).unwrap();
        let term2 = one_minus_y.mul(&log_one_minus_p).unwrap();

        let loss = term1.add(&term2).unwrap();
        let neg_loss = loss.mul_scalar(-1.0);

        match self.reduction {
            Reduction::None => neg_loss,
            Reduction::Sum => {
                let sum = neg_loss.sum();
                Tensor::from_slice(&[sum], &[1]).unwrap()
            }
            Reduction::Mean => {
                let mean = neg_loss.mean();
                Tensor::from_slice(&[mean], &[1]).unwrap()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BCEWithLogitsLoss {
    reduction: Reduction,
}

impl BCEWithLogitsLoss {
    pub fn new() -> Self {
        Self::with_reduction(Reduction::Mean)
    }
    
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl Default for BCEWithLogitsLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for BCEWithLogitsLoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let zeros = Tensor::zeros(&prediction.shape());
        let max_val = prediction.data().mapv(|x| x.max(0.0));
        let max_tensor = Tensor::from_array(max_val);

        let neg_abs_x = prediction.abs().mul_scalar(-1.0);
        let log_sum_exp = neg_abs_x.exp().add_scalar(1.0).log();

        let term1 = max_tensor.sub(&prediction.mul(target).unwrap()).unwrap();
        let loss = term1.add(&log_sum_exp).unwrap();

        match self.reduction {
            Reduction::None => loss,
            Reduction::Sum => {
                let sum = loss.sum();
                Tensor::from_slice(&[sum], &[1]).unwrap()
            }
            Reduction::Mean => {
                let mean = loss.mean();
                Tensor::from_slice(&[mean], &[1]).unwrap()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CrossEntropyLoss {
    reduction: Reduction,
    weight: Option<Tensor>,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self::with_reduction(Reduction::Mean)
    }
    
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self {
            reduction,
            weight: None,
        }
    }
    
    pub fn with_weight(weight: Tensor) -> Self {
        Self {
            reduction: Reduction::Mean,
            weight: Some(weight),
        }
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for CrossEntropyLoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let shape = prediction.shape();
        let num_classes = shape[shape.len() - 1];

        let max_logits = prediction.max();
        let shifted = prediction.add_scalar(-max_logits);
        let exp_shifted = shifted.exp();
        let sum_exp = exp_shifted.sum_dim(shape.len() - 1, true).unwrap();
        let log_sum_exp = sum_exp.log();
        let log_softmax = shifted.sub(&log_sum_exp).unwrap();

        let loss = if target.shape().len() == 1 ||
            (target.shape().len() == 2 && target.shape()[1] == 1) {
            self.nll_loss_indices(&log_softmax, target)
        } else {
            self.nll_loss_onehot(&log_softmax, target)
        };

        match self.reduction {
            Reduction::None => loss,
            Reduction::Sum => {
                let sum = loss.sum();
                Tensor::from_slice(&[sum], &[1]).unwrap()
            }
            Reduction::Mean => {
                let mean = loss.mean();
                Tensor::from_slice(&[mean], &[1]).unwrap()
            }
        }
    }
}

impl CrossEntropyLoss {
    fn nll_loss_indices(&self, log_softmax: &Tensor, target: &Tensor) -> Tensor {
        let batch_size = log_softmax.shape()[0];
        let mut losses = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let target_idx = target.get(&[i]).unwrap_or(0.0) as usize;
            let log_prob = log_softmax.get(&[i, target_idx]).unwrap_or(0.0);
            losses.push(-log_prob);
        }

        Tensor::from_slice(&losses, &[batch_size]).unwrap()
    }

    fn nll_loss_onehot(&self, log_softmax: &Tensor, target: &Tensor) -> Tensor {
        let weighted = log_softmax.mul(target).unwrap();
        let neg_weighted = weighted.mul_scalar(-1.0);
        neg_weighted.sum_dim(neg_weighted.ndim() - 1, false).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct NLLLoss {
    reduction: Reduction,
    weight: Option<Tensor>,
}

impl NLLLoss {
    pub fn new() -> Self {
        Self::with_reduction(Reduction::Mean)
    }
    
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self {
            reduction,
            weight: None,
        }
    }
}

impl Default for NLLLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for NLLLoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let batch_size = prediction.shape()[0];
        let mut losses = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let target_idx = target.get(&[i]).unwrap_or(0.0) as usize;
            let log_prob = prediction.get(&[i, target_idx]).unwrap_or(0.0);
            losses.push(-log_prob);
        }

        let loss = Tensor::from_slice(&losses, &[batch_size]).unwrap();

        match self.reduction {
            Reduction::None => loss,
            Reduction::Sum => {
                let sum = loss.sum();
                Tensor::from_slice(&[sum], &[1]).unwrap()
            }
            Reduction::Mean => {
                let mean = loss.mean();
                Tensor::from_slice(&[mean], &[1]).unwrap()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct HingeLoss {
    margin: TensorDtype,
    reduction: Reduction,
}

impl HingeLoss {
    pub fn new() -> Self {
        Self::with_params(1.0, Reduction::Mean)
    }
    
    pub fn with_params(margin: TensorDtype, reduction: Reduction) -> Self {
        Self { margin, reduction }
    }
}

impl Default for HingeLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for HingeLoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let margin = prediction.mul(target).unwrap().mul_scalar(-1.0).add_scalar(self.margin);
        let zeros = Tensor::zeros(&margin.shape());
        let loss = margin.max_value(&zeros).unwrap();

        match self.reduction {
            Reduction::None => loss,
            Reduction::Sum => {
                let sum = loss.sum();
                Tensor::from_slice(&[sum], &[1]).unwrap()
            }
            Reduction::Mean => {
                let mean = loss.mean();
                Tensor::from_slice(&[mean], &[1]).unwrap()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct HuberLoss {
    delta: TensorDtype,
    reduction: Reduction,
}

impl HuberLoss {
    pub fn new() -> Self {
        Self::with_params(1.0, Reduction::Mean)
    }
    
    pub fn with_params(delta: TensorDtype, reduction: Reduction) -> Self {
        Self { delta, reduction }
    }
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for HuberLoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let diff = prediction.sub(target).unwrap();
        let abs_diff = diff.abs();

        let loss_data: Vec<TensorDtype> = diff.data()
            .iter()
            .zip(abs_diff.data().iter())
            .map(|(&d, &ad)| {
                if ad < self.delta {
                    0.5 * d * d
                } else {
                    self.delta * ad - 0.5 * self.delta * self.delta
                }
            })
            .collect();

        let loss = Tensor::from_slice(&loss_data, &diff.shape()).unwrap();

        match self.reduction {
            Reduction::None => loss,
            Reduction::Sum => {
                let sum = loss.sum();
                Tensor::from_slice(&[sum], &[1]).unwrap()
            }
            Reduction::Mean => {
                let mean = loss.mean();
                Tensor::from_slice(&[mean], &[1]).unwrap()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct KLDivLoss {
    reduction: Reduction,
}

impl KLDivLoss {
    pub fn new() -> Self {
        Self::with_reduction(Reduction::Mean)
    }
    
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl Default for KLDivLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for KLDivLoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let log_pred = prediction.log();
        let diff = target.mul(&log_pred).unwrap();
        let loss = diff.mul_scalar(-1.0);

        match self.reduction {
            Reduction::None => loss,
            Reduction::Sum => {
                let sum = loss.sum();
                Tensor::from_slice(&[sum], &[1]).unwrap()
            }
            Reduction::Mean => {
                let mean = loss.mean();
                Tensor::from_slice(&[mean], &[1]).unwrap()
            }
        }
    }
}