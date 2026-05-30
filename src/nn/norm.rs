use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::{Tensor, TensorDtype};
use ndarray::{ArrayD, Axis, IxDyn};

pub struct InstanceNorm2d {
    pub num_features: usize,
    pub eps: TensorDtype,
    pub momentum: TensorDtype,
    pub affine: bool,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub running_mean: Tensor,
    pub running_var: Tensor,
}

impl InstanceNorm2d {
    pub fn new(num_features: usize) -> Self {
        let weight = Some(Tensor::ones(&[num_features]));
        let bias = Some(Tensor::zeros(&[num_features]));
        let running_mean = Tensor::zeros(&[num_features]);
        let running_var = Tensor::ones(&[num_features]);

        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            weight,
            bias,
            running_mean,
            running_var,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        
        let mut output = Vec::new();
        for b in 0..batch {
            for c in 0..channels {
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                let count = (height * width) as TensorDtype;
                
                for h in 0..height {
                    for w in 0..width {
                        let val = input.get(&[b, c, h, w]);
                        sum += val;
                        sum_sq += val * val;
                    }
                }
                
                let mean = sum / count;
                let var = sum_sq / count - mean * mean;
                let std = (var + self.eps).sqrt();
                
                for h in 0..height {
                    for w in 0..width {
                        let val = input.get(&[b, c, h, w]);
                        let normalized = (val - mean) / std;
                        
                        let scaled = if self.affine {
                            let w = self.weight.as_ref().map(|w| w.get(&[c])).unwrap_or(1.0);
                            let b = self.bias.as_ref().map(|b| b.get(&[c])).unwrap_or(0.0);
                            normalized * w + b
                        } else {
                            normalized
                        };
                        output.push(scaled);
                    }
                }
            }
        }
        
        Tensor::from_slice(&output, &shape).expect("instance norm")
    }
}

impl Module for InstanceNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        if let Some(ref w) = self.weight {
            params.push(Variable::from_tensor(w.clone()));
        }
        if let Some(ref b) = self.bias {
            params.push(Variable::from_tensor(b.clone()));
        }
        params
    }
}

pub struct GroupNorm {
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: TensorDtype,
    pub affine: bool,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        assert!(num_channels % num_groups == 0, "num_channels must be divisible by num_groups");
        
        let weight = Some(Tensor::ones(&[num_channels]));
        let bias = Some(Tensor::zeros(&[num_channels]));

        Self {
            num_groups,
            num_channels,
            eps: 1e-5,
            affine: true,
            weight,
            bias,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let channels_per_group = channels / self.num_groups;
        
        let mut output = Vec::new();
        for b in 0..batch {
            for g in 0..self.num_groups {
                let c_start = g * channels_per_group;
                let c_end = c_start + channels_per_group;
                
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                let count = (channels_per_group * height * width) as TensorDtype;
                
                for c in c_start..c_end {
                    for h in 0..height {
                        for w in 0..width {
                            let val = input.get(&[b, c, h, w]);
                            sum += val;
                            sum_sq += val * val;
                        }
                    }
                }
                
                let mean = sum / count;
                let var = sum_sq / count - mean * mean;
                let std = (var + self.eps).sqrt();
                
                for c in c_start..c_end {
                    for h in 0..height {
                        for w in 0..width {
                            let val = input.get(&[b, c, h, w]);
                            let normalized = (val - mean) / std;
                            
                            let scaled = if self.affine {
                                let w = self.weight.as_ref().map(|w| w.get(&[c])).unwrap_or(1.0);
                                let b = self.bias.as_ref().map(|b| b.get(&[c])).unwrap_or(0.0);
                                normalized * w + b
                            } else {
                                normalized
                            };
                            output.push(scaled);
                        }
                    }
                }
            }
        }
        
        Tensor::from_slice(&output, &shape).expect("group norm")
    }
}

impl Module for GroupNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        if let Some(ref w) = self.weight {
            params.push(Variable::from_tensor(w.clone()));
        }
        if let Some(ref b) = self.bias {
            params.push(Variable::from_tensor(b.clone()));
        }
        params
    }
}
