use crate::tensor::{Tensor, TensorDtype};
use rand::distributions::{Distribution, Uniform};
use rand_distr::StandardNormal;

pub fn xavier_uniform_(tensor: &mut Tensor, gain: f64) {
    let shape = tensor.shape();
    let fan_in = shape.iter().skip(1).product::<usize>().max(1);
    let fan_out = shape[0];
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt() * gain;
    
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(-limit, limit);
    let data: Vec<TensorDtype> = (0..tensor.numel())
        .map(|_| dist.sample(&mut rng) as TensorDtype)
        .collect();
    *tensor = Tensor::from_slice(&data, &shape).expect("xavier uniform");
}

pub fn xavier_normal_(tensor: &mut Tensor, gain: f64) {
    let shape = tensor.shape();
    let fan_in = shape.iter().skip(1).product::<usize>().max(1);
    let fan_out = shape[0];
    let std = gain * (2.0 / (fan_in + fan_out) as f64).sqrt();
    
    let mut rng = rand::thread_rng();
    let dist = StandardNormal;
    let data: Vec<TensorDtype> = (0..tensor.numel())
        .map(|_| (dist.sample(&mut rng) * std) as TensorDtype)
        .collect();
    *tensor = Tensor::from_slice(&data, &shape).expect("xavier normal");
}

pub fn kaiming_uniform_(tensor: &mut Tensor, a: f64, mode: &str, nonlinearity: &str) {
    let shape = tensor.shape();
    let fan = if mode == "fan_in" {
        shape.iter().skip(1).product::<usize>().max(1)
    } else {
        shape[0]
    };
    
    let gain = calculate_gain(nonlinearity, a);
    let bound = (3.0_f64).sqrt() * gain / (fan as f64).sqrt();
    
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(-bound, bound);
    let data: Vec<TensorDtype> = (0..tensor.numel())
        .map(|_| dist.sample(&mut rng) as TensorDtype)
        .collect();
    *tensor = Tensor::from_slice(&data, &shape).expect("kaiming uniform");
}

pub fn kaiming_normal_(tensor: &mut Tensor, a: f64, mode: &str, nonlinearity: &str) {
    let shape = tensor.shape();
    let fan = if mode == "fan_in" {
        shape.iter().skip(1).product::<usize>().max(1)
    } else {
        shape[0]
    };
    
    let gain = calculate_gain(nonlinearity, a);
    let std = gain / (fan as f64).sqrt();
    
    let mut rng = rand::thread_rng();
    let dist = StandardNormal;
    let data: Vec<TensorDtype> = (0..tensor.numel())
        .map(|_| (dist.sample(&mut rng) * std) as TensorDtype)
        .collect();
    *tensor = Tensor::from_slice(&data, &shape).expect("kaiming normal");
}

pub fn orthogonal_(tensor: &mut Tensor, gain: f64) {
    let shape = tensor.shape();
    if shape.len() < 2 {
        return;
    }
    
    let rows = shape[0];
    let cols = shape.iter().skip(1).product();
    let flattened = rows.max(cols);
    
    let mut rng = rand::thread_rng();
    let dist = StandardNormal;
    let mut data: Vec<TensorDtype> = (0..flattened * flattened)
        .map(|_| dist.sample(&mut rng) as TensorDtype)
        .collect();
    
    let q = gram_schmidt(&mut data, flattened);
    
    let final_data: Vec<TensorDtype> = if rows < cols {
        q.iter().take(rows * cols).cloned().collect()
    } else {
        q.iter().take(rows * cols).cloned().collect()
    };
    
    *tensor = Tensor::from_slice(&final_data, &shape).expect("orthogonal");
    *tensor = tensor.mul(&Tensor::full(&[1], gain as TensorDtype));
}

fn gram_schmidt(data: &mut [TensorDtype], n: usize) -> Vec<TensorDtype> {
    let mut q = vec![0.0; n * n];
    
    for i in 0..n {
        for j in 0..n {
            q[i * n + j] = data[i * n + j];
        }
        
        for k in 0..i {
            let mut dot = 0.0;
            for j in 0..n {
                dot += q[k * n + j] * data[i * n + j];
            }
            for j in 0..n {
                q[i * n + j] -= dot * q[k * n + j];
            }
        }
        
        let mut norm = 0.0;
        for j in 0..n {
            norm += q[i * n + j] * q[i * n + j];
        }
        norm = norm.sqrt();
        
        if norm > 1e-8 {
            for j in 0..n {
                q[i * n + j] /= norm;
            }
        }
    }
    
    q
}

fn calculate_gain(nonlinearity: &str, param: f64) -> f64 {
    match nonlinearity {
        "linear" | "conv1d" | "conv2d" | "conv3d" | "conv_transpose1d" | "conv_transpose2d" | "conv_transpose3d" | "sigmoid" => 1.0,
        "tanh" => 5.0 / 3.0,
        "relu" => (2.0_f64).sqrt(),
        "leaky_relu" => (2.0 / (1.0 + param * param)).sqrt(),
        _ => 1.0,
    }
}

pub fn uniform_(tensor: &mut Tensor, low: f64, high: f64) {
    let shape = tensor.shape();
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(low, high);
    let data: Vec<TensorDtype> = (0..tensor.numel())
        .map(|_| dist.sample(&mut rng) as TensorDtype)
        .collect();
    *tensor = Tensor::from_slice(&data, &shape).expect("uniform");
}

pub fn normal_(tensor: &mut Tensor, mean: f64, std: f64) {
    let shape = tensor.shape();
    let mut rng = rand::thread_rng();
    let dist = StandardNormal;
    let data: Vec<TensorDtype> = (0..tensor.numel())
        .map(|_| (dist.sample(&mut rng) * std + mean) as TensorDtype)
        .collect();
    *tensor = Tensor::from_slice(&data, &shape).expect("normal");
}

pub fn constant_(tensor: &mut Tensor, val: TensorDtype) {
    let shape = tensor.shape();
    let data = vec![val; tensor.numel()];
    *tensor = Tensor::from_slice(&data, &shape).expect("constant");
}

pub fn ones_(tensor: &mut Tensor) {
    constant_(tensor, 1.0);
}

pub fn zeros_(tensor: &mut Tensor) {
    constant_(tensor, 0.0);
}

pub fn eye_(tensor: &mut Tensor) {
    let shape = tensor.shape();
    assert!(shape.len() == 2 && shape[0] == shape[1], "eye_ requires square 2D tensor");
    let n = shape[0];
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    *tensor = Tensor::from_slice(&data, &shape).expect("eye");
}
