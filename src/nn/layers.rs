//! Neural network layer implementations

use ndarray::{ArrayD, Axis, IxDyn};

use crate::tensor::{Tensor, TensorDtype};
use crate::nn::{Module, calculate_output_size, xavier_uniform, kaiming_uniform, zeros_init};

/// Linear (fully connected) layer
///
/// Applies a linear transformation: y = xW^T + b
#[derive(Debug)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    /// Create a new Linear layer
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = xavier_uniform(&[out_features, in_features], in_features, out_features);
        let bias = zeros_init(&[out_features]);
        
        let mut weight = weight;
        weight.set_requires_grad(true);
        let mut bias = bias;
        bias.set_requires_grad(true);
        
        Self {
            weight,
            bias: Some(bias),
            in_features,
            out_features,
        }
    }

    /// Create a Linear layer without bias
    pub fn new_without_bias(in_features: usize, out_features: usize) -> Self {
        let weight = xavier_uniform(&[out_features, in_features], in_features, out_features);
        let mut weight = weight;
        weight.set_requires_grad(true);
        
        Self {
            weight,
            bias: None,
            in_features,
            out_features,
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch_size, in_features]
        // weight: [out_features, in_features]
        // output: [batch_size, out_features]
        
        let output = input.matmul(&self.weight.transpose(0, 1).unwrap()).unwrap();
        
        if let Some(ref bias) = self.bias {
            // Add bias: broadcast across batch dimension
            let bias_expanded = bias.unsqueeze(0).unwrap();
            output.add(&bias_expanded).unwrap()
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

/// 2D Convolutional layer
///
/// Applies a 2D convolution over an input signal composed of several input planes.
#[derive(Debug)]
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
}

impl Conv2d {
    /// Create a new Conv2d layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Self {
        Self::new_with_params(
            in_channels,
            out_channels,
            kernel_size,
            (1, 1), // stride
            (0, 0), // padding
            (1, 1), // dilation
            true,   // use_bias
        )
    }

    /// Create a Conv2d layer with custom parameters
    pub fn new_with_params(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        use_bias: bool,
    ) -> Self {
        // Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let fan_out = out_channels * kernel_size.0 * kernel_size.1;
        let weight = kaiming_uniform(
            &[out_channels, in_channels, kernel_size.0, kernel_size.1],
            fan_in,
        );
        
        let mut weight = weight;
        weight.set_requires_grad(true);
        
        let bias = if use_bias {
            let mut bias = zeros_init(&[out_channels]);
            bias.set_requires_grad(true);
            Some(bias)
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input shape: [batch_size, in_channels, height, width]
        // weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        // output shape: [batch_size, out_channels, out_h, out_w]
        
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            panic!("Conv2d expects 4D input [N, C, H, W], got {:?}", input_shape);
        }
        
        let batch_size = input_shape[0];
        let in_height = input_shape[2];
        let in_width = input_shape[3];
        
        let out_height = calculate_output_size(
            in_height,
            self.kernel_size.0,
            self.stride.0,
            self.padding.0,
            self.dilation.0,
        );
        let out_width = calculate_output_size(
            in_width,
            self.kernel_size.1,
            self.stride.1,
            self.padding.1,
            self.dilation.1,
        );
        
        // Perform convolution (naive implementation)
        let output = conv2d_naive(
            input,
            &self.weight,
            self.stride,
            self.padding,
            self.dilation,
        );
        
        if let Some(ref bias) = self.bias {
            // Add bias: reshape to [1, out_channels, 1, 1] for broadcasting
            let bias_reshaped = bias.reshape(&[1, self.out_channels, 1, 1]).unwrap();
            output.add(&bias_reshaped).unwrap()
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

/// Naive 2D convolution implementation
fn conv2d_naive(
    input: &Tensor,
    weight: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Tensor {
    let input_shape = input.shape();
    let weight_shape = weight.shape();
    
    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];
    
    let out_channels = weight_shape[0];
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];
    
    let out_height = calculate_output_size(in_height, kernel_h, stride.0, padding.0, dilation.0);
    let out_width = calculate_output_size(in_width, kernel_w, stride.1, padding.1, dilation.1);
    
    // Initialize output tensor
    let mut output = Tensor::zeros(&[batch_size, out_channels, out_height, out_width]);
    
    // Get data slices for faster access
    let input_data = input.data();
    let weight_data = weight.data();
    let output_data = output.data_mut();
    
    // Perform convolution
    for n in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut sum = 0.0;
                    
                    for ic in 0..in_channels {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = oh * stride.0 + kh * dilation.0;
                                let iw = ow * stride.1 + kw * dilation.1;
                                
                                // Apply padding
                                let ih_padded = ih as i64 - padding.0 as i64;
                                let iw_padded = iw as i64 - padding.1 as i64;
                                
                                if ih_padded >= 0 && ih_padded < in_height as i64 &&
                                   iw_padded >= 0 && iw_padded < in_width as i64 {
                                    let input_val = input_data[[n, ic, ih_padded as usize, iw_padded as usize]];
                                    let weight_val = weight_data[[oc, ic, kh, kw]];
                                    sum += input_val * weight_val;
                                }
                            }
                        }
                    }
                    
                    output_data[[n, oc, oh, ow]] = sum;
                }
            }
        }
    }
    
    output
}

/// Batch Normalization layer
#[derive(Debug)]
pub struct BatchNorm2d {
    pub num_features: usize,
    pub eps: TensorDtype,
    pub momentum: TensorDtype,
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub training: bool,
}

impl BatchNorm2d {
    /// Create a new BatchNorm2d layer
    pub fn new(num_features: usize) -> Self {
        Self::with_params(num_features, 1e-5, 0.1)
    }

    /// Create with custom parameters
    pub fn with_params(num_features: usize, eps: TensorDtype, momentum: TensorDtype) -> Self {
        let weight = Tensor::ones(&[num_features]);
        let bias = Tensor::zeros(&[num_features]);
        let running_mean = Tensor::zeros(&[num_features]);
        let running_var = Tensor::ones(&[num_features]);
        
        let mut weight = weight;
        weight.set_requires_grad(true);
        let mut bias = bias;
        bias.set_requires_grad(true);
        
        Self {
            num_features,
            eps,
            momentum,
            weight,
            bias,
            running_mean,
            running_var,
            training: true,
        }
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input shape: [N, C, H, W]
        let shape = input.shape();
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        
        if self.training {
            // Compute batch statistics
            let mean = input.mean_dim(1, true).unwrap(); // Mean over N, H, W
            let var = input.pow(2.0).mean_dim(1, true).unwrap(); // Simplified variance
            
            // Normalize
            let normalized = input.sub(&mean).unwrap();
            let normalized = normalized.div(&var.sqrt().add_scalar(self.eps)).unwrap();
            
            // Scale and shift
            let weight_reshaped = self.weight.reshape(&[1, c, 1, 1]).unwrap();
            let bias_reshaped = self.bias.reshape(&[1, c, 1, 1]).unwrap();
            
            let scaled = normalized.mul(&weight_reshaped).unwrap();
            scaled.add(&bias_reshaped).unwrap()
        } else {
            // Use running statistics
            let mean_reshaped = self.running_mean.reshape(&[1, c, 1, 1]).unwrap();
            let var_reshaped = self.running_var.reshape(&[1, c, 1, 1]).unwrap();
            
            let normalized = input.sub(&mean_reshaped).unwrap();
            let normalized = normalized.div(&var_reshaped.sqrt().add_scalar(self.eps)).unwrap();
            
            let weight_reshaped = self.weight.reshape(&[1, c, 1, 1]).unwrap();
            let bias_reshaped = self.bias.reshape(&[1, c, 1, 1]).unwrap();
            
            let scaled = normalized.mul(&weight_reshaped).unwrap();
            scaled.add(&bias_reshaped).unwrap()
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

/// Layer Normalization
#[derive(Debug)]
pub struct LayerNorm {
    pub normalized_shape: Vec<usize>,
    pub eps: TensorDtype,
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LayerNorm {
    /// Create a new LayerNorm layer
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        Self::with_eps(normalized_shape, 1e-5)
    }

    /// Create with custom epsilon
    pub fn with_eps(normalized_shape: Vec<usize>, eps: TensorDtype) -> Self {
        let num_features: usize = normalized_shape.iter().product();
        let weight = Tensor::ones(&[num_features]);
        let bias = Tensor::zeros(&[num_features]);
        
        let mut weight = weight;
        weight.set_requires_grad(true);
        let mut bias = bias;
        bias.set_requires_grad(true);
        
        Self {
            normalized_shape,
            eps,
            weight,
            bias,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Compute mean and var over the last len(normalized_shape) dimensions
        let ndim = input.ndim();
        let start_dim = ndim - self.normalized_shape.len();
        
        // Flatten the normalized dimensions
        let batch_dims: Vec<usize> = input.shape()[..start_dim].to_vec();
        let batch_size: usize = batch_dims.iter().product::<usize>().max(1);
        let normalized_size: usize = self.normalized_shape.iter().product();
        
        let flattened = input.reshape(&[batch_size, normalized_size]).unwrap();
        
        // Compute statistics along dimension 1
        let mean = flattened.mean_dim(1, true).unwrap();
        let var = flattened.pow(2.0).mean_dim(1, true).unwrap();
        
        // Normalize
        let normalized = flattened.sub(&mean).unwrap();
        let normalized = normalized.div(&var.sqrt().add_scalar(self.eps)).unwrap();
        
        // Scale and shift
        let weight_reshaped = self.weight.reshape(&[1, normalized_size]).unwrap();
        let bias_reshaped = self.bias.reshape(&[1, normalized_size]).unwrap();
        
        let scaled = normalized.mul(&weight_reshaped).unwrap();
        let output = scaled.add(&bias_reshaped).unwrap();
        
        // Reshape back
        let mut output_shape = batch_dims;
        output_shape.extend(&self.normalized_shape);
        output.reshape(&output_shape).unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

/// Dropout layer
#[derive(Debug)]
pub struct Dropout {
    pub p: TensorDtype,
    pub training: bool,
}

impl Dropout {
    /// Create a new Dropout layer
    pub fn new(p: TensorDtype) -> Self {
        assert!((0.0..=1.0).contains(&p), "Dropout probability must be in [0, 1]");
        Self { p, training: true }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }
        
        if self.p == 1.0 {
            return Tensor::zeros(&input.shape());
        }
        
        // Generate random mask
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.p);
        
        let mask_data: Vec<TensorDtype> = input.data()
            .iter()
            .map(|_| if rng.gen::<TensorDtype>() > self.p { scale } else { 0.0 })
            .collect();
        
        let mask = Tensor::from_slice(&mask_data, &input.shape()).unwrap();
        input.mul(&mask).unwrap()
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

/// Max Pooling 2D layer
#[derive(Debug)]
pub struct MaxPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl MaxPool2d {
    /// Create a new MaxPool2d layer
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: (0, 0),
        }
    }

    /// Create with custom stride
    pub fn with_stride(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride,
            padding: (0, 0),
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];
        
        let out_height = calculate_output_size(
            in_height,
            self.kernel_size.0,
            self.stride.0,
            self.padding.0,
            1,
        );
        let out_width = calculate_output_size(
            in_width,
            self.kernel_size.1,
            self.stride.1,
            self.padding.1,
            1,
        );
        
        let mut output = Tensor::zeros(&[batch_size, channels, out_height, out_width]);
        let input_data = input.data();
        let output_data = output.data_mut();
        
        for n in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut max_val = TensorDtype::NEG_INFINITY;
                        
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = oh * self.stride.0 + kh;
                                let iw = ow * self.stride.1 + kw;
                                
                                if ih < in_height && iw < in_width {
                                    let val = input_data[[n, c, ih, iw]];
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                        }
                        
                        output_data[[n, c, oh, ow]] = max_val;
                    }
                }
            }
        }
        
        output
    }
}

/// Average Pooling 2D layer
#[derive(Debug)]
pub struct AvgPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl AvgPool2d {
    /// Create a new AvgPool2d layer
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: (0, 0),
        }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];
        
        let out_height = calculate_output_size(
            in_height,
            self.kernel_size.0,
            self.stride.0,
            self.padding.0,
            1,
        );
        let out_width = calculate_output_size(
            in_width,
            self.kernel_size.1,
            self.stride.1,
            self.padding.1,
            1,
        );
        
        let mut output = Tensor::zeros(&[batch_size, channels, out_height, out_width]);
        let input_data = input.data();
        let output_data = output.data_mut();
        
        let kernel_area = (self.kernel_size.0 * self.kernel_size.1) as TensorDtype;
        
        for n in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0;
                        let mut count = 0;
                        
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = oh * self.stride.0 + kh;
                                let iw = ow * self.stride.1 + kw;
                                
                                if ih < in_height && iw < in_width {
                                    sum += input_data[[n, c, ih, iw]];
                                    count += 1;
                                }
                            }
                        }
                        
                        output_data[[n, c, oh, ow]] = sum / count as TensorDtype;
                    }
                }
            }
        }
        
        output
    }
}

/// Flatten layer
#[derive(Debug)]
pub struct Flatten;

impl Flatten {
    /// Create a new Flatten layer
    pub fn new() -> Self {
        Self
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        let batch_size = input.shape()[0];
        let flat_size: usize = input.shape()[1..].iter().product();
        input.reshape(&[batch_size, flat_size]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        let linear = Linear::new(3, 2);
        let input = Tensor::ones(&[1, 3]);
        let output = linear.forward(&input);
        assert_eq!(output.shape(), vec![1, 2]);
    }

    #[test]
    fn test_conv2d() {
        let conv = Conv2d::new(3, 16, (3, 3));
        let input = Tensor::ones(&[1, 3, 32, 32]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), vec![1, 16, 30, 30]);
    }

    #[test]
    fn test_batch_norm() {
        let bn = BatchNorm2d::new(16);
        let input = Tensor::ones(&[2, 16, 8, 8]);
        let output = bn.forward(&input);
        assert_eq!(output.shape(), vec![2, 16, 8, 8]);
    }

    #[test]
    fn test_max_pool() {
        let pool = MaxPool2d::new((2, 2));
        let input = Tensor::ones(&[1, 3, 32, 32]);
        let output = pool.forward(&input);
        assert_eq!(output.shape(), vec![1, 3, 16, 16]);
    }
}
