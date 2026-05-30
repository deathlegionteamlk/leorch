use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::{Tensor, TensorDtype};
use rand::distributions::Distribution;
use rand_distr::StandardNormal;

pub struct Embedding {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub padding_idx: Option<usize>,
    pub max_norm: Option<TensorDtype>,
    pub norm_type: f32,
    pub scale_grad_by_freq: bool,
    pub sparse: bool,
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let dist = StandardNormal;
        let std = 1.0 / (embedding_dim as TensorDtype).sqrt();
        let data: Vec<TensorDtype> = (0..num_embeddings * embedding_dim)
            .map(|_| dist.sample(&mut rng) as TensorDtype * std)
            .collect();
        let weight = Tensor::from_slice(&data, &[num_embeddings, embedding_dim]).expect("embedding weight");

        Self {
            num_embeddings,
            embedding_dim,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
            weight,
        }
    }

    pub fn with_padding_idx(mut self, padding_idx: usize) -> Self {
        self.padding_idx = Some(padding_idx);
        self
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();
        let flat_input = input.flatten();
        let indices = flat_input.to_vec();
        
        let mut output_data = Vec::new();
        for &idx in &indices {
            let idx_usize = idx as usize;
            if idx_usize < self.num_embeddings {
                let row = self.weight.slice(&[idx_usize..idx_usize+1, 0..self.embedding_dim]);
                output_data.extend(row.to_vec());
            } else {
                output_data.extend(vec![0.0; self.embedding_dim]);
            }
        }

        let mut output_shape = input_shape.clone();
        output_shape.push(self.embedding_dim);
        Tensor::from_slice(&output_data, &output_shape).expect("embedding output")
    }

    pub fn from_pretrained(weight: Tensor, freeze: bool) -> Self {
        let shape = weight.shape();
        let num_embeddings = shape[0];
        let embedding_dim = shape[1];
        
        Self {
            num_embeddings,
            embedding_dim,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
            weight: if freeze { weight.detach() } else { weight },
        }
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}
