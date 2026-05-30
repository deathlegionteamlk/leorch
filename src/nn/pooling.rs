use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::{Tensor, TensorDtype};

pub struct AdaptiveAvgPool2d {
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let (out_h, out_w) = self.output_size;
        
        let stride_h = height / out_h;
        let stride_w = width / out_w;
        let kernel_h = height - (out_h - 1) * stride_h;
        let kernel_w = width - (out_w - 1) * stride_w;
        
        let mut output = Vec::new();
        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let h_start = oh * stride_h;
                        let w_start = ow * stride_w;
                        let mut sum = 0.0;
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let h = h_start + kh;
                                let w = w_start + kw;
                                let val = input.get(&[b, c, h, w]);
                                sum += val;
                            }
                        }
                        output.push(sum / (kernel_h * kernel_w) as TensorDtype);
                    }
                }
            }
        }
        
        Tensor::from_slice(&output, &[batch, channels, out_h, out_w]).expect("adaptive avg pool")
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct AdaptiveMaxPool2d {
    pub output_size: (usize, usize),
    pub return_indices: bool,
}

impl AdaptiveMaxPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size, return_indices: false }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let (out_h, out_w) = self.output_size;
        
        let stride_h = height / out_h;
        let stride_w = width / out_w;
        let kernel_h = height - (out_h - 1) * stride_h;
        let kernel_w = width - (out_w - 1) * stride_w;
        
        let mut output = Vec::new();
        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let h_start = oh * stride_h;
                        let w_start = ow * stride_w;
                        let mut max_val = TensorDtype::MIN;
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let h = h_start + kh;
                                let w = w_start + kw;
                                let val = input.get(&[b, c, h, w]);
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                        output.push(max_val);
                    }
                }
            }
        }
        
        Tensor::from_slice(&output, &[batch, channels, out_h, out_w]).expect("adaptive max pool")
    }
}

impl Module for AdaptiveMaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![]
    }
}
