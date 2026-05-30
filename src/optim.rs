use crate::tensor::{Tensor, TensorDtype};
use std::collections::HashMap;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
    fn lr(&self) -> TensorDtype;
    fn set_lr(&mut self, lr: TensorDtype);
    fn state_dict(&self) -> HashMap<String, Vec<Tensor>>;
    fn load_state_dict(&mut self, state_dict: HashMap<String, Vec<Tensor>>);
}

#[derive(Debug, Clone)]
pub struct SGD {
    params: Vec<Tensor>,
    lr: TensorDtype,
    momentum: TensorDtype,
    weight_decay: TensorDtype,
    dampening: TensorDtype,
    nesterov: bool,
    velocities: Vec<Option<Tensor>>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: TensorDtype) -> Self {
        Self::with_options(params, lr, 0.0, 0.0, 0.0, false)
    }
    
    pub fn with_options(
        params: Vec<Tensor>,
        lr: TensorDtype,
        momentum: TensorDtype,
        weight_decay: TensorDtype,
        dampening: TensorDtype,
        nesterov: bool,
    ) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            momentum,
            weight_decay,
            dampening,
            nesterov,
            velocities: vec![None; n],
        }
    }
    
    pub fn with_momentum(params: Vec<Tensor>, lr: TensorDtype, momentum: TensorDtype) -> Self {
        Self::with_options(params, lr, momentum, 0.0, 0.0, false)
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (i, param) in self.params.iter_mut().enumerate() {
            let mut grad = if self.weight_decay != 0.0 {
                param.mul_scalar(self.weight_decay)
            } else {
                Tensor::zeros(&param.shape())
            };
            
            if let Some(velocity) = self.velocities[i].as_ref() {
                let momentum_term = velocity.mul_scalar(self.momentum);
                let dampening_term = grad.mul_scalar(1.0 - self.dampening);
                let new_velocity = momentum_term.add(&dampening_term).unwrap();
                self.velocities[i] = Some(new_velocity.clone());
                
                if self.nesterov {
                    let nesterov_grad = grad.add(&new_velocity.mul_scalar(self.momentum)).unwrap();
                    grad = nesterov_grad;
                } else {
                    grad = new_velocity;
                }
            } else {
                self.velocities[i] = Some(grad.clone());
            }
            
            let update = grad.mul_scalar(self.lr);
            let new_param = param.sub(&update).unwrap();
            *param = new_param;
        }
    }
    
    fn zero_grad(&mut self) {
        for param in &mut self.params {
        }
    }
    
    fn lr(&self) -> TensorDtype {
        self.lr
    }
    
    fn set_lr(&mut self, lr: TensorDtype) {
        self.lr = lr;
    }
    
    fn state_dict(&self) -> HashMap<String, Vec<Tensor>> {
        let mut state = HashMap::new();
        state.insert("params".to_string(), self.params.clone());
        state.insert("velocities".to_string(), self.velocities.iter().filter_map(|v| v.clone()).collect());
        state
    }
    
    fn load_state_dict(&mut self, state_dict: HashMap<String, Vec<Tensor>>) {
        if let Some(params) = state_dict.get("params") {
            self.params = params.clone();
        }
    }
}

#[derive(Debug, Clone)]
pub struct Adam {
    params: Vec<Tensor>,
    lr: TensorDtype,
    betas: (TensorDtype, TensorDtype),
    eps: TensorDtype,
    weight_decay: TensorDtype,
    amsgrad: bool,
    t: usize,
    m: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
    v_max: Vec<Option<Tensor>>,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: TensorDtype) -> Self {
        Self::with_options(params, lr, (0.9, 0.999), 1e-8, 0.0, false)
    }
    
    pub fn with_options(
        params: Vec<Tensor>,
        lr: TensorDtype,
        betas: (TensorDtype, TensorDtype),
        eps: TensorDtype,
        weight_decay: TensorDtype,
        amsgrad: bool,
    ) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            t: 0,
            m: vec![None; n],
            v: vec![None; n],
            v_max: vec![None; n],
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
        let beta1 = self.betas.0;
        let beta2 = self.betas.1;
        let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.t as i32);
        
        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = if self.weight_decay != 0.0 {
                param.mul_scalar(self.weight_decay)
            } else {
                Tensor::zeros(&param.shape())
            };
            
            let m = self.m[i].get_or_insert_with(|| Tensor::zeros(&param.shape()));
            let v = self.v[i].get_or_insert_with(|| Tensor::zeros(&param.shape()));
            
            let m_new = m.mul_scalar(beta1).add(&grad.mul_scalar(1.0 - beta1)).unwrap();
            let v_new = v.mul_scalar(beta2).add(&grad.pow(2.0).mul_scalar(1.0 - beta2)).unwrap();
            
            self.m[i] = Some(m_new.clone());
            self.v[i] = Some(v_new.clone());
            
            let m_hat = m_new.mul_scalar(1.0 / bias_correction1);
            let v_hat = if self.amsgrad {
                let v_max = self.v_max[i].get_or_insert_with(|| Tensor::zeros(&param.shape()));
                let v_max_new = Tensor::from_slice(&[v_max.max().max(v_new.max())], &[1]).unwrap();
                self.v_max[i] = Some(v_max_new.clone());
                v_max_new.mul_scalar(1.0 / bias_correction2)
            } else {
                v_new.mul_scalar(1.0 / bias_correction2)
            };
            
            let denom = v_hat.sqrt().add_scalar(self.eps);
            let step_size = m_hat.div(&denom).unwrap().mul_scalar(self.lr);
            let new_param = param.sub(&step_size).unwrap();
            *param = new_param;
        }
    }
    
    fn zero_grad(&mut self) {
        for param in &mut self.params {
        }
    }
    
    fn lr(&self) -> TensorDtype {
        self.lr
    }
    
    fn set_lr(&mut self, lr: TensorDtype) {
        self.lr = lr;
    }
    
    fn state_dict(&self) -> HashMap<String, Vec<Tensor>> {
        let mut state = HashMap::new();
        state.insert("params".to_string(), self.params.clone());
        state.insert("m".to_string(), self.m.iter().filter_map(|v| v.clone()).collect());
        state.insert("v".to_string(), self.v.iter().filter_map(|v| v.clone()).collect());
        state
    }
    
    fn load_state_dict(&mut self, state_dict: HashMap<String, Vec<Tensor>>) {
        if let Some(params) = state_dict.get("params") {
            self.params = params.clone();
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdamW {
    adam: Adam,
    weight_decay: TensorDtype,
}

impl AdamW {
    pub fn new(params: Vec<Tensor>, lr: TensorDtype) -> Self {
        Self::with_weight_decay(params, lr, 0.01)
    }
    
    pub fn with_weight_decay(params: Vec<Tensor>, lr: TensorDtype, weight_decay: TensorDtype) -> Self {
        let adam = Adam::with_options(params, lr, (0.9, 0.999), 1e-8, 0.0, false);
        Self { adam, weight_decay }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        for param in &mut self.adam.params {
            let decay = param.mul_scalar(self.weight_decay * self.adam.lr);
            let new_param = param.sub(&decay).unwrap();
            *param = new_param;
        }
        self.adam.step();
    }
    
    fn zero_grad(&mut self) {
        self.adam.zero_grad();
    }
    
    fn lr(&self) -> TensorDtype {
        self.adam.lr()
    }
    
    fn set_lr(&mut self, lr: TensorDtype) {
        self.adam.set_lr(lr);
    }
    
    fn state_dict(&self) -> HashMap<String, Vec<Tensor>> {
        self.adam.state_dict()
    }
    
    fn load_state_dict(&mut self, state_dict: HashMap<String, Vec<Tensor>>) {
        self.adam.load_state_dict(state_dict);
    }
}

#[derive(Debug, Clone)]
pub struct RMSprop {
    params: Vec<Tensor>,
    lr: TensorDtype,
    alpha: TensorDtype,
    eps: TensorDtype,
    weight_decay: TensorDtype,
    momentum: TensorDtype,
    centered: bool,
    square_avg: Vec<Option<Tensor>>,
    momentum_buffer: Vec<Option<Tensor>>,
    grad_avg: Vec<Option<Tensor>>,
}

impl RMSprop {
    pub fn new(params: Vec<Tensor>, lr: TensorDtype) -> Self {
        Self::with_options(params, lr, 0.99, 1e-8, 0.0, 0.0, false)
    }
    
    pub fn with_options(
        params: Vec<Tensor>,
        lr: TensorDtype,
        alpha: TensorDtype,
        eps: TensorDtype,
        weight_decay: TensorDtype,
        momentum: TensorDtype,
        centered: bool,
    ) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
            square_avg: vec![None; n],
            momentum_buffer: vec![None; n],
            grad_avg: vec![None; n],
        }
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) {
        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = if self.weight_decay != 0.0 {
                param.mul_scalar(self.weight_decay)
            } else {
                Tensor::zeros(&param.shape())
            };
            
            let square_avg = self.square_avg[i].get_or_insert_with(|| Tensor::zeros(&param.shape()));
            let new_square_avg = square_avg.mul_scalar(self.alpha)
                .add(&grad.pow(2.0).mul_scalar(1.0 - self.alpha))
                .unwrap();
            self.square_avg[i] = Some(new_square_avg.clone());
            
            let avg = if self.centered {
                let grad_avg = self.grad_avg[i].get_or_insert_with(|| Tensor::zeros(&param.shape()));
                let new_grad_avg = grad_avg.mul_scalar(self.alpha)
                    .add(&grad.mul_scalar(1.0 - self.alpha))
                    .unwrap();
                self.grad_avg[i] = Some(new_grad_avg.clone());
                new_square_avg.sub(&new_grad_avg.pow(2.0)).unwrap()
            } else {
                new_square_avg
            };
            
            let denom = avg.sqrt().add_scalar(self.eps);
            let step = grad.div(&denom).unwrap().mul_scalar(self.lr);
            
            let final_step = if self.momentum > 0.0 {
                let buf = self.momentum_buffer[i].get_or_insert_with(|| Tensor::zeros(&param.shape()));
                let new_buf = buf.mul_scalar(self.momentum).add(&step).unwrap();
                self.momentum_buffer[i] = Some(new_buf.clone());
                new_buf
            } else {
                step
            };
            
            let new_param = param.sub(&final_step).unwrap();
            *param = new_param;
        }
    }
    
    fn zero_grad(&mut self) {
        for param in &mut self.params {
        }
    }
    
    fn lr(&self) -> TensorDtype {
        self.lr
    }
    
    fn set_lr(&mut self, lr: TensorDtype) {
        self.lr = lr;
    }
    
    fn state_dict(&self) -> HashMap<String, Vec<Tensor>> {
        let mut state = HashMap::new();
        state.insert("params".to_string(), self.params.clone());
        state.insert("square_avg".to_string(), self.square_avg.iter().filter_map(|v| v.clone()).collect());
        state
    }
    
    fn load_state_dict(&mut self, state_dict: HashMap<String, Vec<Tensor>>) {
        if let Some(params) = state_dict.get("params") {
            self.params = params.clone();
        }
    }
}

pub trait LRScheduler {
    fn step(&mut self);
    fn get_last_lr(&self) -> TensorDtype;
}

pub struct StepLR {
optimizer: Box<dyn Optimizer>,
    step_size: usize,
    gamma: TensorDtype,
    last_epoch: usize,
    base_lr: TensorDtype,
    }

impl StepLR {
pub fn new(optimizer: Box<dyn Optimizer>, step_size: usize, gamma: TensorDtype) -> Self {
    let base_lr = optimizer.lr();
        Self {
        optimizer,
            step_size,
            gamma,
            last_epoch: 0,
            base_lr,
            }
        }
    }

impl LRScheduler for StepLR {
fn step(&mut self) {
    self.last_epoch += 1;
        if self.last_epoch % self.step_size == 0 {
        let new_lr = self.base_lr * self.gamma.powi((self.last_epoch / self.step_size) as i32);
            self.optimizer.set_lr(new_lr);
            }
        }
    
    fn get_last_lr(&self) -> TensorDtype {
    self.optimizer.lr()
        }
    }

pub struct ExponentialLR {
optimizer: Box<dyn Optimizer>,
gamma: TensorDtype,
    last_epoch: usize,
    base_lr: TensorDtype,
    }
    
impl ExponentialLR {
pub fn new(optimizer: Box<dyn Optimizer>, gamma: TensorDtype) -> Self {
let base_lr = optimizer.lr();
    Self {
        optimizer,
        gamma,
            last_epoch: 0,
            base_lr,
            }
            }
        }
    
impl LRScheduler for ExponentialLR {
fn step(&mut self) {
self.last_epoch += 1;
    let new_lr = self.base_lr * self.gamma.powi(self.last_epoch as i32);
        self.optimizer.set_lr(new_lr);
        }
        
    fn get_last_lr(&self) -> TensorDtype {
    self.optimizer.lr()
    }
        }
    
pub struct CosineAnnealingLR {
optimizer: Box<dyn Optimizer>,
t_max: usize,
eta_min: TensorDtype,
    last_epoch: usize,
    base_lr: TensorDtype,
    }
    
    impl CosineAnnealingLR {
pub fn new(optimizer: Box<dyn Optimizer>, t_max: usize, eta_min: TensorDtype) -> Self {
let base_lr = optimizer.lr();
Self {
    optimizer,
        t_max,
        eta_min,
            last_epoch: 0,
            base_lr,
            }
            }
            }
        
    impl LRScheduler for CosineAnnealingLR {
fn step(&mut self) {
self.last_epoch += 1;
let t = self.last_epoch as TensorDtype;
    let t_max = self.t_max as TensorDtype;
        let cos_inner = std::f32::consts::PI * t / t_max;
        let new_lr = self.eta_min + (self.base_lr - self.eta_min) * (1.0 + cos_inner.cos()) / 2.0;
        self.optimizer.set_lr(new_lr);
        }
        
        fn get_last_lr(&self) -> TensorDtype {
    self.optimizer.lr()
    }
    }

pub struct ReduceLROnPlateau {
optimizer: Box<dyn Optimizer>,
    mode: String,
    factor: TensorDtype,
    patience: usize,
    threshold: TensorDtype,
    cooldown: usize,
    min_lr: TensorDtype,
    best: TensorDtype,
    num_bad_epochs: usize,
    cooldown_counter: usize,
    base_lr: TensorDtype,
    }

impl ReduceLROnPlateau {
    pub fn new(optimizer: Box<dyn Optimizer>, mode: &str, factor: TensorDtype, patience: usize) -> Self {
        let base_lr = optimizer.lr();
        let best = if mode == "min" { TensorDtype::INFINITY } else { TensorDtype::NEG_INFINITY };
        Self {
            optimizer,
            mode: mode.to_string(),
            factor,
            patience,
            threshold: 1e-4,
            cooldown: 0,
            min_lr: 0.0,
            best,
            num_bad_epochs: 0,
            cooldown_counter: 0,
            base_lr,
        }
    }
    
    pub fn step_with_metric(&mut self, metric: TensorDtype) {
        let improved = if self.mode == "min" {
            metric < self.best * (1.0 - self.threshold)
        } else {
            metric > self.best * (1.0 + self.threshold)
        };
        
        if improved {
            self.best = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }
        
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            self.num_bad_epochs = 0;
        }
        
        if self.num_bad_epochs > self.patience {
            let current_lr = self.optimizer.lr();
            if current_lr > self.min_lr {
                let new_lr = (current_lr * self.factor).max(self.min_lr);
                self.optimizer.set_lr(new_lr);
                self.cooldown_counter = self.cooldown;
                self.num_bad_epochs = 0;
            }
        }
    }
}

impl LRScheduler for ReduceLROnPlateau {
    fn step(&mut self) {
    }
    
    fn get_last_lr(&self) -> TensorDtype {
        self.optimizer.lr()
    }
}