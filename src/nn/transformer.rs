use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::{Tensor, TensorDtype};
use ndarray::{ArrayD, Axis, IxDyn};
use std::f32::consts::PI;

pub struct MultiheadAttention {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub dropout: f32,
    pub bias: bool,
    pub add_bias_kv: bool,
    pub add_zero_attn: bool,
    pub kdim: usize,
    pub vdim: usize,
    pub batch_first: bool,
    pub in_proj_weight: Tensor,
    pub in_proj_bias: Option<Tensor>,
    pub out_proj_weight: Tensor,
    pub out_proj_bias: Option<Tensor>,
    pub bias_k: Option<Tensor>,
    pub bias_v: Option<Tensor>,
}

impl MultiheadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");
        
        let head_dim = embed_dim / num_heads;
        let in_proj_weight = Self::xavier_uniform(&[embed_dim * 3, embed_dim]);
        let out_proj_weight = Self::xavier_uniform(&[embed_dim, embed_dim]);
        
        Self {
            embed_dim,
            num_heads,
            dropout: 0.0,
            bias: true,
            add_bias_kv: false,
            add_zero_attn: false,
            kdim: embed_dim,
            vdim: embed_dim,
            batch_first: false,
            in_proj_weight,
            in_proj_bias: Some(Tensor::zeros(&[embed_dim * 3])),
            out_proj_weight,
            out_proj_bias: Some(Tensor::zeros(&[embed_dim])),
            bias_k: None,
            bias_v: None,
        }
    }

    fn xavier_uniform(shape: &[usize]) -> Tensor {
        let fan_in = shape[1];
        let fan_out = shape[0];
        let limit = (6.0 / (fan_in + fan_out) as TensorDtype).sqrt();
        let mut rng = rand::thread_rng();
        use rand::distributions::Uniform;
        let dist = Uniform::new(-limit, limit);
        let data: Vec<TensorDtype> = (0..shape[0] * shape[1])
            .map(|_| dist.sample(&mut rng))
            .collect();
        Tensor::from_slice(&data, shape).expect("xavier init")
    }

    fn softmax(x: &Tensor, dim: usize) -> Tensor {
        let max_val = x.max_dim(dim, true);
        let exp_x = x.sub(&max_val).exp();
        let sum_exp = exp_x.sum_dim(dim, true);
        exp_x.div(&sum_exp)
    }

    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor, key_padding_mask: Option<&Tensor>, need_weights: bool, attn_mask: Option<&Tensor>) -> (Tensor, Option<Tensor>) {
        let (tgt_len, bsz, embed_dim) = if self.batch_first {
            (query.shape()[1], query.shape()[0], query.shape()[2])
        } else {
            (query.shape()[0], query.shape()[1], query.shape()[2])
        };

        let head_dim = embed_dim / self.num_heads;
        let scale = 1.0 / (head_dim as TensorDtype).sqrt();

        let qkv = self.in_proj_weight.matmul(&query.reshape(&[embed_dim, tgt_len * bsz]));
        if let Some(ref bias) = self.in_proj_bias {
            let _ = qkv.add(bias);
        }

        let qkv_split = qkv.split(&[embed_dim; 3], 0);
        let q = qkv_split[0].reshape(&[embed_dim, tgt_len, bsz]).transpose(0, 1).transpose(1, 2);
        let k = qkv_split[1].reshape(&[embed_dim, tgt_len, bsz]).transpose(0, 1).transpose(1, 2);
        let v = qkv_split[2].reshape(&[embed_dim, tgt_len, bsz]).transpose(0, 1).transpose(1, 2);

        let q = q.reshape(&[bsz * self.num_heads, tgt_len, head_dim]);
        let k = k.reshape(&[bsz * self.num_heads, tgt_len, head_dim]);
        let v = v.reshape(&[bsz * self.num_heads, tgt_len, head_dim]);

        let q_scaled = q.mul(&Tensor::full(&[1], scale));
        let attn_weights = q_scaled.matmul(&k.transpose(1, 2));

        let attn_weights = if let Some(mask) = attn_mask {
            attn_weights.add(mask)
        } else {
            attn_weights
        };

        let attn_weights = Self::softmax(&attn_weights, 2);
        let attn_output = attn_weights.matmul(&v);

        let attn_output = attn_output.reshape(&[bsz, self.num_heads, tgt_len, head_dim]);
        let attn_output = attn_output.transpose(1, 2).reshape(&[bsz, tgt_len, embed_dim]);

        let output = self.out_proj_weight.matmul(&attn_output.transpose(1, 2).reshape(&[embed_dim, tgt_len * bsz]));
        let output = if let Some(ref bias) = self.out_proj_bias {
            output.add(bias)
        } else {
            output
        };

        let output = output.reshape(&[embed_dim, tgt_len, bsz]).transpose(0, 1).transpose(1, 2);

        if self.batch_first {
            (output.transpose(0, 1), None)
        } else {
            (output, None)
        }
    }
}

impl Module for MultiheadAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input, input, input, None, false, None).0
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.in_proj_weight.clone(),
            self.out_proj_weight.clone(),
        ];
        if let Some(ref bias) = self.in_proj_bias {
            params.push(bias.clone());
        }
        if let Some(ref bias) = self.out_proj_bias {
            params.push(bias.clone());
        }
        params
    }
}

pub struct TransformerEncoderLayer {
    pub d_model: usize,
    pub nhead: usize,
    pub dim_feedforward: usize,
    pub dropout: f32,
    pub activation: String,
    pub layer_norm_eps: f32,
    pub batch_first: bool,
    pub norm_first: bool,
    pub self_attn: MultiheadAttention,
    pub linear1: crate::nn::layers::Linear,
    pub linear2: crate::nn::layers::Linear,
    pub norm1: crate::nn::layers::BatchNorm2d,
    pub norm2: crate::nn::layers::BatchNorm2d,
    pub dropout1: crate::nn::layers::Dropout,
    pub dropout2: crate::nn::layers::Dropout,
}

impl TransformerEncoderLayer {
    pub fn new(d_model: usize, nhead: usize) -> Self {
        let dim_feedforward = 2048;
        let self_attn = MultiheadAttention::new(d_model, nhead);
        let linear1 = crate::nn::layers::Linear::new(d_model, dim_feedforward);
        let linear2 = crate::nn::layers::Linear::new(dim_feedforward, d_model);
        let norm1 = crate::nn::layers::BatchNorm2d::new(d_model);
        let norm2 = crate::nn::layers::BatchNorm2d::new(d_model);
        let dropout1 = crate::nn::layers::Dropout::new(0.1);
        let dropout2 = crate::nn::layers::Dropout::new(0.1);

        Self {
            d_model,
            nhead,
            dim_feedforward,
            dropout: 0.1,
            activation: "relu".to_string(),
            layer_norm_eps: 1e-5,
            batch_first: false,
            norm_first: false,
            self_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            dropout1,
            dropout2,
        }
    }

    pub fn forward(&self, src: &Tensor, src_mask: Option<&Tensor>, is_causal: bool) -> Tensor {
        let x = if self.norm_first {
            let x_norm = self.norm1.forward(src);
            let attn_output = self.self_attn.forward(&x_norm, &x_norm, &x_norm, None, false, src_mask).0;
            let x = src.add(&self.dropout1.forward(&attn_output));
            let x_norm2 = self.norm2.forward(&x);
            let ff_output = self.linear2.forward(&self.activation(&self.linear1.forward(&x_norm2)));
            x.add(&self.dropout2.forward(&ff_output))
        } else {
            let attn_output = self.self_attn.forward(src, src, src, None, false, src_mask).0;
            let x = self.norm1.forward(&src.add(&self.dropout1.forward(&attn_output)));
            let ff_output = self.linear2.forward(&self.activation(&self.linear1.forward(&x)));
            self.norm2.forward(&x.add(&self.dropout2.forward(&ff_output)))
        };
        x
    }

    fn activation(&self, x: &Tensor) -> Tensor {
        match self.activation.as_str() {
            "relu" => x.relu(),
            "gelu" => x.gelu(),
            _ => x.relu(),
        }
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input, None, false)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.self_attn.parameters();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }
}

pub struct TransformerDecoderLayer {
    pub d_model: usize,
    pub nhead: usize,
    pub dim_feedforward: usize,
    pub dropout: f32,
    pub activation: String,
    pub layer_norm_eps: f32,
    pub batch_first: bool,
    pub norm_first: bool,
    pub self_attn: MultiheadAttention,
    pub multihead_attn: MultiheadAttention,
    pub linear1: crate::nn::layers::Linear,
    pub linear2: crate::nn::layers::Linear,
    pub norm1: crate::nn::layers::BatchNorm2d,
    pub norm2: crate::nn::layers::BatchNorm2d,
    pub norm3: crate::nn::layers::BatchNorm2d,
    pub dropout1: crate::nn::layers::Dropout,
    pub dropout2: crate::nn::layers::Dropout,
    pub dropout3: crate::nn::layers::Dropout,
}

impl TransformerDecoderLayer {
    pub fn new(d_model: usize, nhead: usize) -> Self {
        let dim_feedforward = 2048;
        let self_attn = MultiheadAttention::new(d_model, nhead);
        let multihead_attn = MultiheadAttention::new(d_model, nhead);
        let linear1 = crate::nn::layers::Linear::new(d_model, dim_feedforward);
        let linear2 = crate::nn::layers::Linear::new(dim_feedforward, d_model);
        let norm1 = crate::nn::layers::BatchNorm2d::new(d_model);
        let norm2 = crate::nn::layers::BatchNorm2d::new(d_model);
        let norm3 = crate::nn::layers::BatchNorm2d::new(d_model);
        let dropout1 = crate::nn::layers::Dropout::new(0.1);
        let dropout2 = crate::nn::layers::Dropout::new(0.1);
        let dropout3 = crate::nn::layers::Dropout::new(0.1);

        Self {
            d_model,
            nhead,
            dim_feedforward,
            dropout: 0.1,
            activation: "relu".to_string(),
            layer_norm_eps: 1e-5,
            batch_first: false,
            norm_first: false,
            self_attn,
            multihead_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            norm3,
            dropout1,
            dropout2,
            dropout3,
        }
    }

    pub fn forward(&self, tgt: &Tensor, memory: &Tensor, tgt_mask: Option<&Tensor>, memory_mask: Option<&Tensor>, tgt_is_causal: bool, memory_is_causal: bool) -> Tensor {
        let x = if self.norm_first {
            let x_norm = self.norm1.forward(tgt);
            let self_attn = self.self_attn.forward(&x_norm, &x_norm, &x_norm, None, false, tgt_mask).0;
            let x = tgt.add(&self.dropout1.forward(&self_attn));
            
            let x_norm2 = self.norm2.forward(&x);
            let cross_attn = self.multihead_attn.forward(&x_norm2, memory, memory, None, false, memory_mask).0;
            let x = x.add(&self.dropout2.forward(&cross_attn));
            
            let x_norm3 = self.norm3.forward(&x);
            let ff_output = self.linear2.forward(&self.activation(&self.linear1.forward(&x_norm3)));
            x.add(&self.dropout3.forward(&ff_output))
        } else {
            let self_attn = self.self_attn.forward(tgt, tgt, tgt, None, false, tgt_mask).0;
            let x = self.norm1.forward(&tgt.add(&self.dropout1.forward(&self_attn)));
            
            let cross_attn = self.multihead_attn.forward(&x, memory, memory, None, false, memory_mask).0;
            let x = self.norm2.forward(&x.add(&self.dropout2.forward(&cross_attn)));
            
            let ff_output = self.linear2.forward(&self.activation(&self.linear1.forward(&x)));
            self.norm3.forward(&x.add(&self.dropout3.forward(&ff_output)))
        };
        x
    }

    fn activation(&self, x: &Tensor) -> Tensor {
        match self.activation.as_str() {
            "relu" => x.relu(),
            "gelu" => x.gelu(),
            _ => x.relu(),
        }
    }
}

impl Module for TransformerDecoderLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input, input, None, None, false, false)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.self_attn.parameters();
        params.extend(self.multihead_attn.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.norm3.parameters());
        params
    }
}
