pub mod layers;
pub mod activations;
pub mod rnn;
pub mod embedding;
pub mod transformer;
pub mod pooling;
pub mod norm;
pub mod init;

pub use layers::*;
pub use activations::*;
pub use rnn::*;
pub use embedding::*;
pub use transformer::*;
pub use pooling::*;
pub use norm::*;
pub use init::*;

use crate::tensor::{Tensor, TensorDtype};
use crate::autograd::Variable;

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    
    fn train(&mut self) {}
    
    fn eval(&mut self) {}
    
    fn zero_grad(&mut self) {
        for param in self.parameters() {
        }
    }
}

pub trait ModuleForward {
    fn forward(&self, input: &Tensor) -> Tensor;
}

impl<T: Module> ModuleForward for T {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }
}

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: vec![] }
    }
    
    pub fn add(&mut self, layer: Box<dyn Module>) {
        self.layers.push(layer);
    }
    
    pub fn from_layers(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![];
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
    
    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }
    
    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }
}

pub fn calculate_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
}

pub fn xavier_uniform(shape: &[usize], fan_in: usize, fan_out: usize) -> Tensor {
    let limit: TensorDtype = (6.0 / (fan_in + fan_out) as TensorDtype).sqrt();
    let data: Vec<TensorDtype> = (0..shape.iter().product::<usize>())
        .map(|_| rand::random::<TensorDtype>() * 2.0 * limit - limit)
        .collect();
    Tensor::from_slice(&data, shape).expect("Failed to create tensor")
}

pub fn kaiming_uniform(shape: &[usize], fan_in: usize) -> Tensor {
    let limit: TensorDtype = (3.0 / fan_in as TensorDtype).sqrt();
    let data: Vec<TensorDtype> = (0..shape.iter().product::<usize>())
        .map(|_| rand::random::<TensorDtype>() * 2.0 * limit - limit)
        .collect();
    Tensor::from_slice(&data, shape).expect("Failed to create tensor")
}

pub fn zeros_init(shape: &[usize]) -> Tensor {
    Tensor::zeros(shape)
}

pub fn uniform_init(shape: &[usize], low: TensorDtype, high: TensorDtype) -> Tensor {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let data: Vec<TensorDtype> = (0..shape.iter().product::<usize>())
        .map(|_| rng.gen_range(low..high))
        .collect();
    Tensor::from_slice(&data, shape).expect("Failed to create tensor")
}