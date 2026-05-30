use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::{Tensor, TensorDtype};
use ndarray::{ArrayD, Axis, IxDyn};
use rand::distributions::Distribution;
use rand_distr::StandardNormal;
use std::f32;

pub struct RNN {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub nonlinearity: String,
    pub batch_first: bool,
    pub dropout: f32,
    pub bidirectional: bool,
    pub weight_ih_l: Vec<Tensor>,
    pub weight_hh_l: Vec<Tensor>,
    pub bias_ih_l: Vec<Option<Tensor>>,
    pub bias_hh_l: Vec<Option<Tensor>>,
}

impl RNN {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize, nonlinearity: &str, batch_first: bool, dropout: f32, bidirectional: bool) -> Self {
        let num_directions = if bidirectional { 2 } else { 1 };
        let mut weight_ih_l = Vec::new();
        let mut weight_hh_l = Vec::new();
        let mut bias_ih_l = Vec::new();
        let mut bias_hh_l = Vec::new();

        for layer in 0..num_layers {
            for direction in 0..num_directions {
                let layer_input_size = if layer == 0 { input_size } else { hidden_size * num_directions };
                let suffix = if direction == 1 { "_reverse" } else { "" };
                
                let w_ih = Self::init_weight(&format!("weight_ih_l{}{}", layer, suffix), layer_input_size, hidden_size);
                let w_hh = Self::init_weight(&format!("weight_hh_l{}{}", layer, suffix), hidden_size, hidden_size);
                
                weight_ih_l.push(w_ih);
                weight_hh_l.push(w_hh);
                bias_ih_l.push(Some(Tensor::zeros(&[hidden_size])));
                bias_hh_l.push(Some(Tensor::zeros(&[hidden_size])));
            }
        }

        Self {
            input_size,
            hidden_size,
            num_layers,
            nonlinearity: nonlinearity.to_string(),
            batch_first,
            dropout,
            bidirectional,
            weight_ih_l,
            weight_hh_l,
            bias_ih_l,
            bias_hh_l,
        }
    }

    fn init_weight(name: &str, input_size: usize, hidden_size: usize) -> Tensor {
        let std = (1.0 / hidden_size as TensorDtype).sqrt();
        let mut rng = rand::thread_rng();
        let dist = StandardNormal;
        let data: Vec<TensorDtype> = (0..hidden_size * input_size)
            .map(|_| dist.sample(&mut rng) as TensorDtype * std)
            .collect();
        Tensor::from_slice(&data, &[hidden_size, input_size]).expect(name)
    }

    fn activation(&self, x: &Tensor) -> Tensor {
        match self.nonlinearity.as_str() {
            "tanh" => x.tanh(),
            "relu" => x.relu(),
            _ => x.tanh(),
        }
    }

    pub fn forward(&self, input: &Tensor, hx: Option<&Tensor>) -> (Tensor, Tensor) {
        let (seq_len, batch_size, input_size) = if self.batch_first {
            (input.shape()[1], input.shape()[0], input.shape()[2])
        } else {
            (input.shape()[0], input.shape()[1], input.shape()[2])
        };

        let num_directions = if self.bidirectional { 2 } else { 1 };
        let hx = hx.cloned().unwrap_or_else(|| Tensor::zeros(&[self.num_layers * num_directions, batch_size, self.hidden_size]));

        let mut output_seq = Vec::new();
        let mut h_n = Vec::new();

        for direction in 0..num_directions {
            let is_reverse = direction == 1 && self.bidirectional;
            let mut h_prev = hx.slice(&[direction..direction+1, 0..batch_size, 0..self.hidden_size]).reshape(&[batch_size, self.hidden_size]);

            let mut layer_output = Vec::new();
            let seq_indices: Vec<usize> = if is_reverse {
                (0..seq_len).rev().collect()
            } else {
                (0..seq_len).collect()
            };

            for t in seq_indices {
                let x_t = if self.batch_first {
                    input.slice(&[0..batch_size, t..t+1, 0..input_size]).reshape(&[batch_size, input_size])
                } else {
                    input.slice(&[t..t+1, 0..batch_size, 0..input_size]).reshape(&[batch_size, input_size])
                };

                let mut h_t = self.weight_ih_l[direction].matmul(&x_t.transpose()).transpose();
                h_t = h_t.add(&self.weight_hh_l[direction].matmul(&h_prev.transpose()).transpose());
                
                if let Some(ref b_ih) = self.bias_ih_l[direction] {
                    h_t = h_t.add(b_ih);
                }
                if let Some(ref b_hh) = self.bias_hh_l[direction] {
                    h_t = h_t.add(b_hh);
                }

                h_t = self.activation(&h_t);
                h_prev = h_t.clone();
                layer_output.push(h_t);
            }

            if is_reverse {
                layer_output.reverse();
            }
            h_n.push(h_prev);
            output_seq.push(layer_output);
        }

        let mut final_output = Vec::new();
        for t in 0..seq_len {
            let mut time_step = Vec::new();
            for dir in 0..num_directions {
                time_step.push(output_seq[dir][t].clone());
            }
            let concat_t = if num_directions == 2 {
                Tensor::cat(&[time_step[0].clone(), time_step[1].clone()], 1)
            } else {
                time_step[0].clone()
            };
            final_output.push(concat_t);
        }

        let output = if self.batch_first {
            Tensor::stack(&final_output, 1)
        } else {
            Tensor::stack(&final_output, 0)
        };

        let h_n = Tensor::stack(&h_n, 0);
        (output, h_n)
    }
}

impl Module for RNN {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input, None).0
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for w in &self.weight_ih_l {
            params.push(w.clone());
        }
        for w in &self.weight_hh_l {
            params.push(w.clone());
        }
        for b in &self.bias_ih_l {
            if let Some(ref bias) = b {
                params.push(bias.clone());
            }
        }
        for b in &self.bias_hh_l {
            if let Some(ref bias) = b {
                params.push(bias.clone());
            }
        }
        params
    }
}

pub struct LSTM {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub batch_first: bool,
    pub dropout: f32,
    pub bidirectional: bool,
    pub weight_ih_l: Vec<Tensor>,
    pub weight_hh_l: Vec<Tensor>,
    pub bias_ih_l: Vec<Option<Tensor>>,
    pub bias_hh_l: Vec<Option<Tensor>>,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize, batch_first: bool, dropout: f32, bidirectional: bool) -> Self {
        let num_directions = if bidirectional { 2 } else { 1 };
        let mut weight_ih_l = Vec::new();
        let mut weight_hh_l = Vec::new();
        let mut bias_ih_l = Vec::new();
        let mut bias_hh_l = Vec::new();

        for layer in 0..num_layers {
            for direction in 0..num_directions {
                let layer_input_size = if layer == 0 { input_size } else { hidden_size * num_directions };
                let suffix = if direction == 1 { "_reverse" } else { "" };
                
                let w_ih = Self::init_weight(&format!("weight_ih_l{}{}", layer, suffix), layer_input_size, hidden_size * 4);
                let w_hh = Self::init_weight(&format!("weight_hh_l{}{}", layer, suffix), hidden_size, hidden_size * 4);
                
                weight_ih_l.push(w_ih);
                weight_hh_l.push(w_hh);
                bias_ih_l.push(Some(Tensor::zeros(&[hidden_size * 4])));
                bias_hh_l.push(Some(Tensor::zeros(&[hidden_size * 4])));
            }
        }

        Self {
            input_size,
            hidden_size,
            num_layers,
            batch_first,
            dropout,
            bidirectional,
            weight_ih_l,
            weight_hh_l,
            bias_ih_l,
            bias_hh_l,
        }
    }

    fn init_weight(name: &str, input_size: usize, hidden_size: usize) -> Tensor {
        let std = (1.0 / hidden_size as TensorDtype).sqrt();
        let mut rng = rand::thread_rng();
        let dist = StandardNormal;
        let data: Vec<TensorDtype> = (0..hidden_size * input_size)
            .map(|_| dist.sample(&mut rng) as TensorDtype * std)
            .collect();
        Tensor::from_slice(&data, &[hidden_size, input_size]).expect(name)
    }

    fn sigmoid(x: &Tensor) -> Tensor {
        x.neg().exp().add(&Tensor::ones(&x.shape())).reciprocal()
    }

    pub fn forward(&self, input: &Tensor, hx: Option<(&Tensor, &Tensor)>) -> (Tensor, Tensor, Tensor) {
        let (seq_len, batch_size, input_size) = if self.batch_first {
            (input.shape()[1], input.shape()[0], input.shape()[2])
        } else {
            (input.shape()[0], input.shape()[1], input.shape()[2])
        };

        let num_directions = if self.bidirectional { 2 } else { 1 };
        let h_zeros = Tensor::zeros(&[self.num_layers * num_directions, batch_size, self.hidden_size]);
        let c_zeros = Tensor::zeros(&[self.num_layers * num_directions, batch_size, self.hidden_size]);
        let hx = hx.map(|(h, c)| (h.clone(), c.clone())).unwrap_or((h_zeros, c_zeros));

        let mut output_seq = Vec::new();
        let mut h_n = Vec::new();
        let mut c_n = Vec::new();

        for direction in 0..num_directions {
            let is_reverse = direction == 1 && self.bidirectional;
            let mut h_prev = hx.0.slice(&[direction..direction+1, 0..batch_size, 0..self.hidden_size]).reshape(&[batch_size, self.hidden_size]);
            let mut c_prev = hx.1.slice(&[direction..direction+1, 0..batch_size, 0..self.hidden_size]).reshape(&[batch_size, self.hidden_size]);

            let mut layer_output = Vec::new();
            let seq_indices: Vec<usize> = if is_reverse {
                (0..seq_len).rev().collect()
            } else {
                (0..seq_len).collect()
            };

            for t in seq_indices {
                let x_t = if self.batch_first {
                    input.slice(&[0..batch_size, t..t+1, 0..input_size]).reshape(&[batch_size, input_size])
                } else {
                    input.slice(&[t..t+1, 0..batch_size, 0..input_size]).reshape(&[batch_size, input_size])
                };

                let gates_ih = self.weight_ih_l[direction].matmul(&x_t.transpose()).transpose();
                let gates_hh = self.weight_hh_l[direction].matmul(&h_prev.transpose()).transpose();
                let mut gates = gates_ih.add(&gates_hh);

                if let Some(ref b_ih) = self.bias_ih_l[direction] {
                    gates = gates.add(b_ih);
                }
                if let Some(ref b_hh) = self.bias_hh_l[direction] {
                    gates = gates.add(b_hh);
                }

                let gates_split = gates.split(&[self.hidden_size; 4], 1);
                let i = Self::sigmoid(&gates_split[0]);
                let f = Self::sigmoid(&gates_split[1]);
                let g = gates_split[2].tanh();
                let o = Self::sigmoid(&gates_split[3]);

                c_prev = f.mul(&c_prev).add(&i.mul(&g));
                h_prev = o.mul(&c_prev.tanh());

                layer_output.push(h_prev.clone());
            }

            if is_reverse {
                layer_output.reverse();
            }
            h_n.push(h_prev);
            c_n.push(c_prev);
            output_seq.push(layer_output);
        }

        let mut final_output = Vec::new();
        for t in 0..seq_len {
            let mut time_step = Vec::new();
            for dir in 0..num_directions {
                time_step.push(output_seq[dir][t].clone());
            }
            let concat_t = if num_directions == 2 {
                Tensor::cat(&[time_step[0].clone(), time_step[1].clone()], 1)
            } else {
                time_step[0].clone()
            };
            final_output.push(concat_t);
        }

        let output = if self.batch_first {
            Tensor::stack(&final_output, 1)
        } else {
            Tensor::stack(&final_output, 0)
        };

        let h_n = Tensor::stack(&h_n, 0);
        let c_n = Tensor::stack(&c_n, 0);
        (output, h_n, c_n)
    }
}

impl Module for LSTM {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input, None).0
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for w in &self.weight_ih_l {
            params.push(w.clone());
        }
        for w in &self.weight_hh_l {
            params.push(w.clone());
        }
        for b in &self.bias_ih_l {
            if let Some(ref bias) = b {
                params.push(bias.clone());
            }
        }
        for b in &self.bias_hh_l {
            if let Some(ref bias) = b {
                params.push(bias.clone());
            }
        }
        params
    }
}

pub struct GRU {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub batch_first: bool,
    pub dropout: f32,
    pub bidirectional: bool,
    pub weight_ih_l: Vec<Tensor>,
    pub weight_hh_l: Vec<Tensor>,
    pub bias_ih_l: Vec<Option<Tensor>>,
    pub bias_hh_l: Vec<Option<Tensor>>,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize, batch_first: bool, dropout: f32, bidirectional: bool) -> Self {
        let num_directions = if bidirectional { 2 } else { 1 };
        let mut weight_ih_l = Vec::new();
        let mut weight_hh_l = Vec::new();
        let mut bias_ih_l = Vec::new();
        let mut bias_hh_l = Vec::new();

        for layer in 0..num_layers {
            for direction in 0..num_directions {
                let layer_input_size = if layer == 0 { input_size } else { hidden_size * num_directions };
                let suffix = if direction == 1 { "_reverse" } else { "" };
                
                let w_ih = Self::init_weight(&format!("weight_ih_l{}{}", layer, suffix), layer_input_size, hidden_size * 3);
                let w_hh = Self::init_weight(&format!("weight_hh_l{}{}", layer, suffix), hidden_size, hidden_size * 3);
                
                weight_ih_l.push(w_ih);
                weight_hh_l.push(w_hh);
                bias_ih_l.push(Some(Tensor::zeros(&[hidden_size * 3])));
                bias_hh_l.push(Some(Tensor::zeros(&[hidden_size * 3])));
            }
        }

        Self {
            input_size,
            hidden_size,
            num_layers,
            batch_first,
            dropout,
            bidirectional,
            weight_ih_l,
            weight_hh_l,
            bias_ih_l,
            bias_hh_l,
        }
    }

    fn init_weight(name: &str, input_size: usize, hidden_size: usize) -> Tensor {
        let std = (1.0 / hidden_size as TensorDtype).sqrt();
        let mut rng = rand::thread_rng();
        let dist = StandardNormal;
        let data: Vec<TensorDtype> = (0..hidden_size * input_size)
            .map(|_| dist.sample(&mut rng) as TensorDtype * std)
            .collect();
        Tensor::from_slice(&data, &[hidden_size, input_size]).expect(name)
    }

    fn sigmoid(x: &Tensor) -> Tensor {
        x.neg().exp().add(&Tensor::ones(&x.shape())).reciprocal()
    }

    pub fn forward(&self, input: &Tensor, hx: Option<&Tensor>) -> (Tensor, Tensor) {
        let (seq_len, batch_size, input_size) = if self.batch_first {
            (input.shape()[1], input.shape()[0], input.shape()[2])
        } else {
            (input.shape()[0], input.shape()[1], input.shape()[2])
        };

        let num_directions = if self.bidirectional { 2 } else { 1 };
        let hx = hx.cloned().unwrap_or_else(|| Tensor::zeros(&[self.num_layers * num_directions, batch_size, self.hidden_size]));

        let mut output_seq = Vec::new();
        let mut h_n = Vec::new();

        for direction in 0..num_directions {
            let is_reverse = direction == 1 && self.bidirectional;
            let mut h_prev = hx.slice(&[direction..direction+1, 0..batch_size, 0..self.hidden_size]).reshape(&[batch_size, self.hidden_size]);

            let mut layer_output = Vec::new();
            let seq_indices: Vec<usize> = if is_reverse {
                (0..seq_len).rev().collect()
            } else {
                (0..seq_len).collect()
            };

            for t in seq_indices {
                let x_t = if self.batch_first {
                    input.slice(&[0..batch_size, t..t+1, 0..input_size]).reshape(&[batch_size, input_size])
                } else {
                    input.slice(&[t..t+1, 0..batch_size, 0..input_size]).reshape(&[batch_size, input_size])
                };

                let gates_ih = self.weight_ih_l[direction].matmul(&x_t.transpose()).transpose();
                let gates_hh = self.weight_hh_l[direction].matmul(&h_prev.transpose()).transpose();
                let mut gates = gates_ih.add(&gates_hh);

                if let Some(ref b_ih) = self.bias_ih_l[direction] {
                    gates = gates.add(b_ih);
                }
                if let Some(ref b_hh) = self.bias_hh_l[direction] {
                    gates = gates.add(b_hh);
                }

                let gates_split = gates.split(&[self.hidden_size; 3], 1);
                let r = Self::sigmoid(&gates_split[0]);
                let z = Self::sigmoid(&gates_split[1]);
                let n = gates_split[2].tanh();

                h_prev = z.neg().add(&Tensor::ones(&z.shape())).mul(&h_prev).add(&z.mul(&n));

                layer_output.push(h_prev.clone());
            }

            if is_reverse {
                layer_output.reverse();
            }
            h_n.push(h_prev);
            output_seq.push(layer_output);
        }

        let mut final_output = Vec::new();
        for t in 0..seq_len {
            let mut time_step = Vec::new();
            for dir in 0..num_directions {
                time_step.push(output_seq[dir][t].clone());
            }
            let concat_t = if num_directions == 2 {
                Tensor::cat(&[time_step[0].clone(), time_step[1].clone()], 1)
            } else {
                time_step[0].clone()
            };
            final_output.push(concat_t);
        }

        let output = if self.batch_first {
            Tensor::stack(&final_output, 1)
        } else {
            Tensor::stack(&final_output, 0)
        };

        let h_n = Tensor::stack(&h_n, 0);
        (output, h_n)
    }
}

impl Module for GRU {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input, None).0
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for w in &self.weight_ih_l {
            params.push(w.clone());
        }
        for w in &self.weight_hh_l {
            params.push(w.clone());
        }
        for b in &self.bias_ih_l {
            if let Some(ref bias) = b {
                params.push(bias.clone());
            }
        }
        for b in &self.bias_hh_l {
            if let Some(ref bias) = b {
                params.push(bias.clone());
            }
        }
        params
    }
}
