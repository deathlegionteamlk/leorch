use crate::tensor::{Tensor, TensorDtype};
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    FP32,
    FP16,
    BF16,
}

impl Default for Precision {
    fn default() -> Self {
        Precision::FP32
    }
}

pub fn to_fp16(tensor: &Tensor) -> Tensor {
    tensor.clone()
}

pub fn to_bf16(tensor: &Tensor) -> Tensor {
    tensor.clone()
}

pub fn to_fp32(tensor: &Tensor) -> Tensor {
    tensor.clone()
}

pub struct GradScaler {
    scale: TensorDtype,
    growth_factor: TensorDtype,
    backoff_factor: TensorDtype,
    growth_interval: usize,
    enabled: bool,
    found_inf: bool,
    _growth_tracker: usize,
}

impl GradScaler {
    pub fn new(enabled: bool) -> Self {
        Self::with_params(enabled, 65536.0, 2.0, 0.5, 2000)
    }
    
    pub fn with_params(
        enabled: bool,
        init_scale: TensorDtype,
        growth_factor: TensorDtype,
        backoff_factor: TensorDtype,
        growth_interval: usize,
    ) -> Self {
        Self {
            scale: init_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            enabled,
            found_inf: false,
            _growth_tracker: 0,
        }
    }
    
    pub fn scale(&self, loss: Tensor) -> Tensor {
        if self.enabled {
            loss.mul_scalar(self.scale)
        } else {
            loss
        }
    }
    
    pub fn unscale(&self, grads: &mut [Tensor]) {
        if !self.enabled {
            return;
        }
        for grad in grads.iter_mut() {
            let unscaled = grad.div_scalar(self.scale);
            *grad = unscaled;
        }
    }
    
    pub fn step(&mut self, optimizer: &mut dyn crate::optim::Optimizer) {
        if self.found_inf {
            self.scale *= self.backoff_factor;
            self._growth_tracker = 0;
        } else {
            optimizer.step();
            self._growth_tracker += 1;
            if self._growth_tracker >= self.growth_interval {
                self.scale *= self.growth_factor;
                self._growth_tracker = 0;
            }
        }
        self.found_inf = false;
    }
    
    pub fn update(&mut self) {
    }
    
    pub fn get_scale(&self) -> TensorDtype {
        self.scale
    }
    
    pub fn set_scale(&mut self, scale: TensorDtype) {
        self.scale = scale;
    }
    
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    pub fn check_inf(&mut self, grads: &[Tensor]) {
        for grad in grads {
            for &val in grad.data().iter() {
                if !val.is_finite() {
                    self.found_inf = true;
                    return;
                }
            }
        }
    }
}

pub struct Autocast {
    enabled: bool,
    precision: Precision,
}

impl Autocast {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            precision: Precision::FP16,
        }
    }
    
    pub fn enabled(&self) -> bool {
        self.enabled
    }
    
    pub fn cast_input(&self, input: &Tensor) -> Tensor {
        if self.enabled {
            match self.precision {
                Precision::FP16 => to_fp16(input),
                Precision::BF16 => to_bf16(input),
                Precision::FP32 => input.clone(),
            }
        } else {
            input.clone()
        }
    }
    
    pub fn cast_output(&self, output: &Tensor) -> Tensor {
        if self.enabled {
            to_fp32(output)
        } else {
            output.clone()
        }
    }
}

pub fn autocast<F, R>(enabled: bool, f: F) -> R
where
    F: FnOnce() -> R,
{
    f()
}
