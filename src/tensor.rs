use ndarray::{Array, ArrayD, Axis, Dimension, IxDyn};
use num_traits::Zero;
use rand::distributions::Distribution;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::error::{LeorchError, Result};

pub type TensorDtype = f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    CPU,
    CUDA(usize),
}

impl Default for Device {
    fn default() -> Self {
        Device::CPU
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub(crate) data: ArrayD<TensorDtype>,
    pub device: Device,
    pub requires_grad: bool,
}

impl Tensor {
    pub fn from_array(data: ArrayD<TensorDtype>) -> Self {
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let data = Array::zeros(IxDyn(shape));
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        let data = Array::ones(IxDyn(shape));
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn full(shape: &[usize], value: TensorDtype) -> Self {
        let data = Array::from_elem(IxDyn(shape), value);
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn eye(n: usize) -> Self {
        let mut data = Array::zeros(IxDyn(&[n, n]));
        for i in 0..n {
            data[[i, i]] = 1.0;
        }
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn logspace(start: TensorDtype, end: TensorDtype, num: usize, base: TensorDtype) -> Self {
        if num == 0 {
            return Self::zeros(&[0]);
        }
        if num == 1 {
            return Self::from_slice(&[base.powf(start)], &[1]).unwrap();
        }
        let step = (end - start) / ((num - 1) as TensorDtype);
        let data: Vec<TensorDtype> = (0..num)
            .map(|i| base.powf(start + step * (i as TensorDtype)))
            .collect();
        let data = Array::from_shape_vec(IxDyn(&[num]), data)
            .expect("Failed to create logspace tensor");
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn meshgrid(x: &Self, y: &Self) -> (Self, Self) {
        let x_shape = x.shape();
        let y_shape = y.shape();
        let x_len = x_shape.iter().product::<usize>();
        let y_len = y_shape.iter().product::<usize>();

        let x_data: Vec<TensorDtype> = x.data.iter().copied().collect();
        let y_data: Vec<TensorDtype> = y.data.iter().copied().collect();

        let mut xx_data = Vec::with_capacity(x_len * y_len);
        let mut yy_data = Vec::with_capacity(x_len * y_len);

        for &yi in &y_data {
            for &xi in &x_data {
                xx_data.push(xi);
                yy_data.push(yi);
            }
        }

        let xx = Self::from_slice(&xx_data, &[y_len, x_len]).unwrap();
        let yy = Self::from_slice(&yy_data, &[y_len, x_len]).unwrap();

        (xx, yy)
    }

    pub fn triu(n: usize) -> Self {
        let mut data = Array::zeros(IxDyn(&[n, n]));
        for i in 0..n {
            for j in i..n {
                data[[i, j]] = 1.0;
            }
        }
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn tril(n: usize) -> Self {
        let mut data = Array::zeros(IxDyn(&[n, n]));
        for i in 0..n {
            for j in 0..=i {
                data[[i, j]] = 1.0;
            }
        }
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn cartesian_prod(vectors: &[&Self]) -> Self {
        if vectors.is_empty() {
            return Self::zeros(&[0, 0]);
        }

        let sizes: Vec<usize> = vectors.iter().map(|v| v.shape().iter().product()).collect();
        let total: usize = sizes.iter().product();
        let num_vectors = vectors.len();

        let mut result_data = Vec::with_capacity(total * num_vectors);

        fn recursive_fill(
            vectors: &[&Tensor],
            indices: &mut Vec<usize>,
            depth: usize,
            result: &mut Vec<TensorDtype>,
        ) {
            if depth == vectors.len() {
                for (i, &idx) in indices.iter().enumerate() {
                    let v: Vec<TensorDtype> = vectors[i].data.iter().copied().collect();
                    result.push(v[idx]);
                }
            } else {
                let len = vectors[depth].shape().iter().product::<usize>();
                for i in 0..len {
                    indices.push(i);
                    recursive_fill(vectors, indices, depth + 1, result);
                    indices.pop();
                }
            }
        }

        let mut indices = Vec::new();
        recursive_fill(vectors, &mut indices, 0, &mut result_data);

        Self::from_slice(&result_data, &[total, num_vectors]).unwrap()
    }

    pub fn randn(shape: &[usize]) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<TensorDtype> = StandardNormal
            .sample_iter(&mut rng)
            .take(shape.iter().product())
            .map(|x: f64| x as TensorDtype)
            .collect();
        let data = Array::from_shape_vec(IxDyn(shape), data)
            .expect("Failed to create tensor from random data");
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn rand(shape: &[usize]) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data: Vec<TensorDtype> = (0..shape.iter().product::<usize>())
            .map(|_| rng.gen::<TensorDtype>())
            .collect();
        let data = Array::from_shape_vec(IxDyn(shape), data).expect("Failed to create rand tensor");
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn normal(mean: TensorDtype, std: TensorDtype, shape: &[usize]) -> Self {
        use rand_distr::Normal;
        let mut rng = rand::thread_rng();
        let dist = Normal::new(mean, std).unwrap();
        let data: Vec<TensorDtype> = (0..shape.iter().product::<usize>())
            .map(|_| dist.sample(&mut rng) as TensorDtype)
            .collect();
        let data = Array::from_shape_vec(IxDyn(shape), data)
            .expect("Failed to create normal tensor");
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn uniform(low: TensorDtype, high: TensorDtype, shape: &[usize]) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data: Vec<TensorDtype> = (0..shape.iter().product::<usize>())
            .map(|_| rng.gen_range(low..high))
            .collect();
        let data = Array::from_shape_vec(IxDyn(shape), data)
            .expect("Failed to create uniform tensor");
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn bernoulli(p: TensorDtype, shape: &[usize]) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data: Vec<TensorDtype> = (0..shape.iter().product::<usize>())
            .map(|_| if rng.gen::<TensorDtype>() < p { 1.0 } else { 0.0 })
            .collect();
        let data = Array::from_shape_vec(IxDyn(shape), data)
            .expect("Failed to create bernoulli tensor");
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn multinomial(probs: &Self, num_samples: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let probs_vec: Vec<TensorDtype> = probs.data.iter().copied().collect();
        let sum: TensorDtype = probs_vec.iter().sum();
        let normalized: Vec<TensorDtype> = probs_vec.iter().map(|&p| p / sum).collect();

        let mut samples = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            let r: TensorDtype = rng.gen();
            let mut cumsum = 0.0;
            let mut idx = 0;
            for (i, &p) in normalized.iter().enumerate() {
                cumsum += p;
                if r <= cumsum {
                    idx = i;
                    break;
                }
            }
            samples.push(idx as TensorDtype);
        }

        Self::from_slice(&samples, &[num_samples]).unwrap()
    }

    pub fn poisson(lambda: TensorDtype, shape: &[usize]) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data: Vec<TensorDtype> = (0..shape.iter().product::<usize>())
            .map(|_| {
                let mut sum = 0.0;
                let mut prod: TensorDtype = rng.gen();
                while prod >= (-lambda).exp() {
                    sum += 1.0;
                    prod *= rng.gen::<TensorDtype>();
                }
                sum
            })
            .collect();
        let data = Array::from_shape_vec(IxDyn(shape), data)
            .expect("Failed to create poisson tensor");
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn from_slice(data: &[TensorDtype], shape: &[usize]) -> Result<Self> {
        if data.len() != shape.iter().product::<usize>() {
            return Err(LeorchError::ShapeMismatch {
                expected: vec![shape.iter().product()],
                got: vec![data.len()],
            });
        }
        let data = Array::from_shape_vec(IxDyn(shape), data.to_vec())
            .map_err(|e| LeorchError::NdarrayError(e.to_string()))?;
        Ok(Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        })
    }

    pub fn arange(start: TensorDtype, end: TensorDtype, step: TensorDtype) -> Self {
        let num = ((end - start) / step).ceil() as usize;
        let data: Vec<TensorDtype> = (0..num)
            .map(|i| start + step * (i as TensorDtype))
            .collect();
        let data = Array::from_shape_vec(IxDyn(&[num]), data)
            .expect("Failed to create arange tensor");
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn linspace(start: TensorDtype, end: TensorDtype, num: usize) -> Self {
        if num == 0 {
            return Self::zeros(&[0]);
        }
        if num == 1 {
            return Self::from_slice(&[start], &[1]).unwrap();
        }
        let step = (end - start) / ((num - 1) as TensorDtype);
        let data: Vec<TensorDtype> = (0..num)
            .map(|i| start + step * (i as TensorDtype))
            .collect();
        let data = Array::from_shape_vec(IxDyn(&[num]), data)
            .expect("Failed to create linspace tensor");
        Self {
            data,
            device: Device::CPU,
            requires_grad: false,
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &ArrayD<TensorDtype> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut ArrayD<TensorDtype> {
        &mut self.data
    }

    pub fn get(&self, indices: &[usize]) -> Option<TensorDtype> {
        self.data.get(indices).copied()
    }

    pub fn set(&mut self, indices: &[usize], value: TensorDtype) -> Result<()> {
        if let Some(elem) = self.data.get_mut(indices) {
            *elem = value;
            Ok(())
        } else {
            Err(LeorchError::IndexOutOfBounds {
                index: indices[0],
                dim: 0,
                size: self.data.shape()[0],
            })
        }
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let new_shape: Vec<usize> = if shape.iter().product::<usize>() == 0 {
            let known_size: usize = shape.iter().filter(|&&x| x != 0).product();
            let inferred = self.numel() / known_size;
            shape.iter().map(|&x| if x == 0 { inferred } else { x }).collect()
        } else {
            shape.to_vec()
        };
        if new_shape.iter().product::<usize>() != self.numel() {
            return Err(LeorchError::ShapeMismatch {
                expected: vec![self.numel()],
                got: vec![new_shape.iter().product()],
            });
        }
        let data = self.data.clone().into_shape(IxDyn(&new_shape))
            .map_err(|e| LeorchError::NdarrayError(e.to_string()))?;
        Ok(Self {
            data,
            device: self.device,
            requires_grad: self.requires_grad,
        })
    }

    pub fn flatten(&self) -> Self {
        self.reshape(&[self.numel()]).unwrap()
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let mut new_shape = self.shape();
        if dim > new_shape.len() {
            return Err(LeorchError::DimensionError(
                format!("Cannot unsqueeze at dim {} for tensor with {} dimensions", dim, new_shape.len())
            ));
        }
        new_shape.insert(dim, 1);
        self.reshape(&new_shape)
    }

    pub fn squeeze(&self, dim: Option<usize>) -> Self {
        match dim {
            Some(d) => {
                let mut new_shape = self.shape();
                if d < new_shape.len() && new_shape[d] == 1 {
                    new_shape.remove(d);
                }
                self.reshape(&new_shape).unwrap_or_else(|_| self.clone())
            }
            None => {
                let new_shape: Vec<usize> = self.shape().into_iter().filter(|&x| x != 1).collect();
                if new_shape.is_empty() {
                    self.reshape(&[1]).unwrap_or_else(|_| self.clone())
                } else {
                    self.reshape(&new_shape).unwrap_or_else(|_| self.clone())
                }
            }
        }
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        if dim0 >= self.ndim() || dim1 >= self.ndim() {
            return Err(LeorchError::DimensionError("Transpose dimensions out of bounds".to_string()));
        }
        let mut axes: Vec<usize> = (0..self.ndim()).collect();
        axes.swap(dim0, dim1);
        let data = self.data.view().permuted_axes(axes.as_slice()).to_owned();
        Ok(Self {
            data,
            device: self.device,
            requires_grad: self.requires_grad,
        })
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        if dims.len() != self.ndim() {
            return Err(LeorchError::DimensionError(
                format!("Permute dimensions {:?} don't match tensor dimensions {}", dims, self.ndim())
            ));
        }
        let data = self.data.view().permuted_axes(dims).to_owned();
        Ok(Self {
            data,
            device: self.device,
            requires_grad: self.requires_grad,
        })
    }

    pub fn cat(tensors: &[&Self], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(LeorchError::InvalidOperation("Cannot concatenate empty list of tensors".to_string()));
        }
        let first_shape = tensors[0].shape();
        for tensor in tensors.iter() {
            if tensor.ndim() != first_shape.len() {
                return Err(LeorchError::ShapeMismatch {
                    expected: first_shape.clone(),
                    got: tensor.shape(),
                });
            }
            for (d, (s1, s2)) in first_shape.iter().zip(tensor.shape().iter()).enumerate() {
                if d != dim && s1 != s2 {
                    return Err(LeorchError::ShapeMismatch {
                        expected: first_shape.clone(),
                        got: tensor.shape(),
                    });
                }
            }
        }
        let arrays: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
        let data = ndarray::concatenate(Axis(dim), &arrays)
            .map_err(|e| LeorchError::NdarrayError(e.to_string()))?;
        Ok(Self {
            data,
            device: tensors[0].device,
            requires_grad: tensors.iter().any(|t| t.requires_grad),
        })
    }

    pub fn stack(tensors: &[&Self], dim: usize) -> Result<Self> {
        let unsqueezed: Result<Vec<_>> = tensors.iter()
            .map(|t| t.unsqueeze(dim))
            .collect();
        Self::cat(&unsqueezed?.iter().collect::<Vec<_>>(), dim)
    }

    pub fn sum(&self) -> TensorDtype {
        self.data.sum()
    }

    pub fn sum_dim(&self, dim: usize, keepdim: bool) -> Result<Self> {
        let data = self.data.sum_axis(Axis(dim));
        let result = Self {
            data,
            device: self.device,
            requires_grad: self.requires_grad,
        };
        if keepdim {
            result.unsqueeze(dim)
        } else {
            Ok(result)
        }
    }

    pub fn mean(&self) -> TensorDtype {
        self.data.mean().unwrap_or(TensorDtype::zero())
    }

    pub fn mean_dim(&self, dim: usize, keepdim: bool) -> Result<Self> {
        let data = self.data.mean_axis(Axis(dim))
            .ok_or_else(|| LeorchError::DimensionError("Cannot compute mean".to_string()))?;
        let result = Self {
            data,
            device: self.device,
            requires_grad: self.requires_grad,
        };
        if keepdim {
            result.unsqueeze(dim)
        } else {
            Ok(result)
        }
    }

    pub fn max(&self) -> TensorDtype {
        self.data.iter().fold(TensorDtype::NEG_INFINITY, |a, &b| a.max(b))
    }

    pub fn min(&self) -> TensorDtype {
        self.data.iter().fold(TensorDtype::INFINITY, |a, &b| a.min(b))
    }

    pub fn max_value(&self, other: &Self) -> Result<Self> {
        let data = ndarray::Zip::from(&self.data).and(&other.data).map_collect(|a, b| a.max(*b));
        Ok(Self {
            data,
            device: self.device,
            requires_grad: self.requires_grad || other.requires_grad,
        })
    }

    pub fn min_value(&self, other: &Self) -> Result<Self> {
        let data = ndarray::Zip::from(&self.data).and(&other.data).map_collect(|a, b| a.min(*b));
        Ok(Self {
            data,
            device: self.device,
            requires_grad: self.requires_grad || other.requires_grad,
        })
    }

    pub fn abs(&self) -> Self {
        Self {
            data: self.data.mapv(|x| x.abs()),
            device: self.device,
            requires_grad: self.requires_grad,
        }
    }

    pub fn sqrt(&self) -> Self {
        Self {
            data: self.data.mapv(|x| x.sqrt()),
            device: self.device,
            requires_grad: self.requires_grad,
        }
    }

    pub fn exp(&self) -> Self {
        Self {
            data: self.data.mapv(|x| x.exp()),
            device: self.device,
            requires_grad: self.requires_grad,
        }
    }

    pub fn log(&self) -> Self {
        Self {
            data: self.data.mapv(|x| x.ln()),
            device: self.device,
            requires_grad: self.requires_grad,
        }
    }

    pub fn pow(&self, exponent: TensorDtype) -> Self {
        Self {
            data: self.data.mapv(|x| x.powf(exponent)),
            device: self.device,
            requires_grad: self.requires_grad,
        }
    }

    pub fn tanh(&self) -> Self {
        Self {
            data: self.data.mapv(|x| x.tanh()),
            device: self.device,
            requires_grad: self.requires_grad,
        }
    }

    pub fn sigmoid(&self) -> Self {
        Self {
            data: self.data.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            device: self.device,
            requires_grad: self.requires_grad,
        }
    }

    pub fn relu(&self) -> Self {
        Self {
            data: self.data.mapv(|x| x.max(0.0)),
            device: self.device,
            requires_grad: self.requires_grad,
        }
    }

    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if self.ndim() < 2 || other.ndim() < 2 {
            return Err(LeorchError::DimensionError(
                "Matmul requires at least 2D tensors".to_string()
            ));
        }
        if self.ndim() == 2 && other.ndim() == 2 {
            let self_view = self.data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            let other_view = other.data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            let result = self_view.dot(&other_view);
            let data = result.into_dyn();
            return Ok(Self {
                data,
                device: self.device,
                requires_grad: self.requires_grad || other.requires_grad,
            });
        }
        Err(LeorchError::DimensionError(
            "Batched matmul not yet implemented".to_string()
        ))
    }

    fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
        let max_len = a.len().max(b.len());
        let mut result = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let dim_a = a.get(a.len().saturating_sub(max_len - i)).copied().unwrap_or(1);
            let dim_b = b.get(b.len().saturating_sub(max_len - i)).copied().unwrap_or(1);
            if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
                return None;
            }
            result.push(dim_a.max(dim_b));
        }
        Some(result)
    }

    fn broadcast_to_shape(data: &ArrayD<TensorDtype>, target_shape: &[usize]) -> ArrayD<TensorDtype> {
        let current_shape: Vec<usize> = data.shape().to_vec();
        if current_shape == target_shape {
            return data.clone();
        }
        let current_numel: usize = current_shape.iter().product();
        let target_numel: usize = target_shape.iter().product();
        if current_numel == target_numel {
            return data.clone().into_shape(IxDyn(target_shape)).unwrap_or_else(|_| data.clone());
        }
        if current_numel == 1 {
            return Array::from_elem(IxDyn(target_shape), data.iter().next().copied().unwrap_or(0.0));
        }
        data.clone()
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        let self_shape = self.shape();
        let other_shape = other.shape();
        let target_shape = Self::broadcast_shapes(&self_shape, &other_shape)
            .ok_or_else(|| LeorchError::ShapeMismatch { expected: self_shape.clone(), got: other_shape.clone() })?;
        let self_broadcasted = Self::broadcast_to_shape(&self.data, &target_shape);
        let other_broadcasted = Self::broadcast_to_shape(&other.data, &target_shape);
        let data = &self_broadcasted * &other_broadcasted;
        Ok(Self {
            data,
            device: self.device,
            requires_grad: self.requires_grad || other.requires_grad,
        })
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        let self_shape = self.shape();
        let other_shape = other.shape();
        let target_shape = Self::broadcast_shapes(&self_shape, &other_shape)
            .ok_or_else(|| LeorchError::ShapeMismatch { expected: self_shape.clone(), got: other_shape.clone() })?;
        let self_broadcasted = Self::broadcast_to_shape(&self.data, &target_shape);
        let other_broadcasted = Self::broadcast_to_shape(&other.data, &target_shape);
        let data = &self_broadcasted + &other_broadcasted;
        Ok(Self {
            data,
            device: self.device,
            requires_grad: self.requires_grad || other.requires_grad,
        })
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        let self_shape = self.shape();
        let other_shape = other.shape();
        let target_shape = Self::broadcast_shapes(&self_shape, &other_shape)
            .ok_or_else(|| LeorchError::ShapeMismatch { expected: self_shape.clone(), got: other_shape.clone() })?;
        let self_broadcasted = Self::broadcast_to_shape(&self.data, &target_shape);
        let other_broadcasted = Self::broadcast_to_shape(&other.data, &target_shape);
        let data = &self_broadcasted - &other_broadcasted;
        Ok(Self {
            data,
            device: self.device,
            requires_grad: self.requires_grad || other.requires_grad,
        })
    }

    pub fn div(&self, other: &Self) -> Result<Self> {
        let self_shape = self.shape();
        let other_shape = other.shape();
        let target_shape = Self::broadcast_shapes(&self_shape, &other_shape)
            .ok_or_else(|| LeorchError::ShapeMismatch { expected: self_shape.clone(), got: other_shape.clone() })?;
        let self_broadcasted = Self::broadcast_to_shape(&self.data, &target_shape);
        let other_broadcasted = Self::broadcast_to_shape(&other.data, &target_shape);
        let data = &self_broadcasted / &other_broadcasted;
        Ok(Self {
            data,
            device: self.device,
            requires_grad: self.requires_grad || other.requires_grad,
        })
    }

    pub fn mul_scalar(&self, scalar: TensorDtype) -> Self {
        Self {
            data: &self.data * scalar,
            device: self.device,
            requires_grad: self.requires_grad,
        }
    }

    pub fn add_scalar(&self, scalar: TensorDtype) -> Self {
        Self {
            data: &self.data + scalar,
            device: self.device,
            requires_grad: self.requires_grad,
        }
    }

    pub fn sub_scalar(&self, scalar: TensorDtype) -> Self {
        Self {
            data: &self.data - scalar,
            device: self.device,
            requires_grad: self.requires_grad,
        }
    }

    pub fn div_scalar(&self, scalar: TensorDtype) -> Self {
        Self {
            data: &self.data / scalar,
            device: self.device,
            requires_grad: self.requires_grad,
        }
    }

    pub fn neg(&self) -> Self {
        Self {
            data: -&self.data,
            device: self.device,
            requires_grad: self.requires_grad,
        }
    }

    pub fn to_vec(&self) -> Vec<TensorDtype> {
        self.data.iter().copied().collect()
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    pub fn slice(&self, index: usize) -> Self {
        let shape = self.shape();
        if shape.is_empty() || index >= shape[0] {
            return Self::zeros(&[0]);
        }
        let row_size: usize = shape[1..].iter().product();
        let start = index * row_size;
        let data_slice: Vec<TensorDtype> = self.data.iter().skip(start).take(row_size).copied().collect();
        let new_shape: Vec<usize> = shape[1..].to_vec();
        Self {
            data: Array::from_shape_vec(IxDyn(&new_shape), data_slice).unwrap_or_else(|_| ArrayD::zeros(IxDyn(&new_shape))),
            device: self.device,
            requires_grad: self.requires_grad,
        }
    }

    pub fn einsum(equation: &str, tensors: &[&Self]) -> Result<Self> {
        if equation == "ij,jk->ik" && tensors.len() == 2 {
            return tensors[0].matmul(tensors[1]);
        }
        if equation == "ij->i" && tensors.len() == 1 {
            return tensors[0].sum_dim(1, false);
        }
        if equation == "ij->j" && tensors.len() == 1 {
            return tensors[0].sum_dim(0, false);
        }
        Err(LeorchError::InvalidOperation(
            format!("Einsum equation '{}' not supported", equation)
        ))
    }

    pub fn tensordot(a: &Self, b: &Self, axes_a: &[usize], axes_b: &[usize]) -> Result<Self> {
        if axes_a.len() == 1 && axes_b.len() == 1 && axes_a[0] == 1 && axes_b[0] == 0 {
            return a.matmul(b);
        }
        Err(LeorchError::InvalidOperation("Tensordot not fully implemented".to_string()))
    }

    pub fn kron(a: &Self, b: &Self) -> Result<Self> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(LeorchError::DimensionError("Kron requires 2D tensors".to_string()));
        }

        let m = a_shape[0] * b_shape[0];
        let n = a_shape[1] * b_shape[1];
        let mut result = Array::zeros(IxDyn(&[m, n]));

        for i in 0..a_shape[0] {
            for j in 0..a_shape[1] {
                let a_val = a.get(&[i, j]).unwrap_or(0.0);
                for k in 0..b_shape[0] {
                    for l in 0..b_shape[1] {
                        let b_val = b.get(&[k, l]).unwrap_or(0.0);
                        result[[i * b_shape[0] + k, j * b_shape[1] + l]] = a_val * b_val;
                    }
                }
            }
        }

        Ok(Self {
            data: result,
            device: a.device,
            requires_grad: a.requires_grad || b.requires_grad,
        })
    }

    pub fn cdist(x1: &Self, x2: &Self, p: TensorDtype) -> Result<Self> {
        let x1_shape = x1.shape();
        let x2_shape = x2.shape();

        if x1_shape.len() != 2 || x2_shape.len() != 2 {
            return Err(LeorchError::DimensionError("cdist requires 2D tensors".to_string()));
        }

        let n = x1_shape[0];
        let m = x2_shape[0];
        let d = x1_shape[1];

        if d != x2_shape[1] {
            return Err(LeorchError::ShapeMismatch {
                expected: vec![n, d],
                got: x2_shape.clone(),
            });
        }

        let mut result = Array::zeros(IxDyn(&[n, m]));

        for i in 0..n {
            for j in 0..m {
                let mut dist = 0.0;
                for k in 0..d {
                    let diff = x1.get(&[i, k]).unwrap_or(0.0) - x2.get(&[j, k]).unwrap_or(0.0);
                    dist += diff.abs().powf(p);
                }
                result[[i, j]] = dist.powf(1.0 / p);
            }
        }

        Ok(Self {
            data: result,
            device: x1.device,
            requires_grad: x1.requires_grad || x2.requires_grad,
        })
    }

    pub fn pdist(x: &Self, p: TensorDtype) -> Result<Self> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(LeorchError::DimensionError("pdist requires 2D tensor".to_string()));
        }

        let n = shape[0];
        let num_pairs = n * (n - 1) / 2;
        let mut result = Array::zeros(IxDyn(&[num_pairs]));

        let mut idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut dist = 0.0;
                for k in 0..shape[1] {
                    let diff = x.get(&[i, k]).unwrap_or(0.0) - x.get(&[j, k]).unwrap_or(0.0);
                    dist += diff.abs().powf(p);
                }
                result[idx] = dist.powf(1.0 / p);
                idx += 1;
            }
        }

        Ok(Self {
            data: result,
            device: x.device,
            requires_grad: x.requires_grad,
        })
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> {
        let self_shape = self.shape();
        if self_shape == shape {
            return Ok(self.clone());
        }

        let self_numel = self_shape.iter().product::<usize>();
        let target_numel = shape.iter().product::<usize>();

        if self_numel == target_numel {
            return self.reshape(shape);
        }

        if self_numel == 1 {
            let data = Array::from_elem(IxDyn(shape), self.data.iter().next().copied().unwrap_or(0.0));
            return Ok(Self {
                data,
                device: self.device,
                requires_grad: self.requires_grad,
            });
        }

        Err(LeorchError::ShapeMismatch {
            expected: shape.to_vec(),
            got: self_shape,
        })
    }

    pub fn index_select(&self, dim: usize, indices: &Self) -> Result<Self> {
        let indices_vec: Vec<usize> = indices.data.iter().map(|&x| x as usize).collect();
        let mut result_shape = self.shape();
        result_shape[dim] = indices_vec.len();

        let mut result_data = Vec::new();
        for &idx in &indices_vec {
            if self.ndim() == 2 {
                for i in 0..self.shape()[1 - dim] {
                    let val = if dim == 0 {
                        self.get(&[idx, i]).unwrap_or(0.0)
                    } else {
                        self.get(&[i, idx]).unwrap_or(0.0)
                    };
                    result_data.push(val);
                }
            }
        }

        Self::from_slice(&result_data, &result_shape)
    }

    pub fn argmax(&self, dim: Option<usize>) -> Result<Self> {
        match dim {
            Some(d) => {
                let axis = Axis(d);
                let data = self.data.map_axis(axis, |view| {
                    view.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as TensorDtype).unwrap_or(0.0)
                });
                Ok(Self {
                    data: data.into_dyn(),
                    device: self.device,
                    requires_grad: false,
                })
            }
            None => {
                let flat = self.data.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as TensorDtype).unwrap_or(0.0);
                Ok(Self::from_slice(&[flat], &[1])?)
            }
        }
    }

    pub fn argmin(&self, dim: Option<usize>) -> Result<Self> {
        match dim {
            Some(d) => {
                let axis = Axis(d);
                let data = self.data.map_axis(axis, |view| {
                    view.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as TensorDtype).unwrap_or(0.0)
                });
                Ok(Self {
                    data: data.into_dyn(),
                    device: self.device,
                    requires_grad: false,
                })
            }
            None => {
                let flat = self.data.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as TensorDtype).unwrap_or(0.0);
                Ok(Self::from_slice(&[flat], &[1])?)
            }
        }
    }

    pub fn prod(&self) -> TensorDtype {
        self.data.iter().fold(1.0, |a, &b| a * b)
    }

    pub fn cumsum(&self, dim: usize) -> Result<Self> {
        let mut result = self.data.clone();
        for mut row in result.axis_iter_mut(Axis(dim)) {
            let mut cumsum = 0.0;
            for elem in row.iter_mut() {
                cumsum += *elem;
                *elem = cumsum;
            }
        }
        Ok(Self {
            data: result,
            device: self.device,
            requires_grad: self.requires_grad,
        })
    }

    pub fn cumprod(&self, dim: usize) -> Result<Self> {
        let mut result = self.data.clone();
        for mut row in result.axis_iter_mut(Axis(dim)) {
            let mut cumprod = 1.0;
            for elem in row.iter_mut() {
                cumprod *= *elem;
                *elem = cumprod;
            }
        }
        Ok(Self {
            data: result,
            device: self.device,
            requires_grad: self.requires_grad,
        })
    }

    pub fn sort(&self, dim: usize, descending: bool) -> Result<(Self, Self)> {
        let axis = Axis(dim);
        let mut values_data = self.data.clone();
        let mut indices_data = Array::from_shape_fn(self.data.raw_dim(), |idx| idx[dim] as TensorDtype);

        for (mut values_row, mut indices_row) in values_data.axis_iter_mut(axis).zip(indices_data.axis_iter_mut(axis)) {
            let mut pairs: Vec<(TensorDtype, TensorDtype)> = values_row.iter().zip(indices_row.iter()).map(|(&v, &i)| (v, i)).collect();
            if descending {
                pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            } else {
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            }
            for (i, (v, idx)) in pairs.iter().enumerate() {
                values_row[i] = *v;
                indices_row[i] = *idx;
            }
        }

        let values = Self {
            data: values_data,
            device: self.device,
            requires_grad: self.requires_grad,
        };
        let indices = Self {
            data: indices_data,
            device: self.device,
            requires_grad: false,
        };
        Ok((values, indices))
    }

    pub fn topk(&self, k: usize, dim: usize, largest: bool) -> Result<(Self, Self)> {
        let (sorted, indices) = self.sort(dim, largest)?;
        let mut result_shape = self.shape();
        result_shape[dim] = k;

        let mut values_data = Vec::new();
        let mut indices_data = Vec::new();

        for idx in 0..self.numel() {
            let coords: Vec<usize> = self.data.raw_dim().as_array_view().iter().enumerate().map(|(d, &s)| {
                if d == dim { idx % k } else { idx % s }
            }).collect();

            if coords[dim] < k {
                if let Some(&val) = sorted.data.get(coords.as_slice()) {
                    values_data.push(val);
                }
                if let Some(&idx_val) = indices.data.get(coords.as_slice()) {
                    indices_data.push(idx_val);
                }
            }
        }

        let values = Self::from_slice(&values_data, &result_shape)?;
        let indices = Self::from_slice(&indices_data, &result_shape)?;
        Ok((values, indices))
    }

    pub fn unique(&self) -> Result<Self> {
        let mut values: Vec<TensorDtype> = self.data.iter().copied().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
        Ok(Self::from_slice(&values, &[values.len()])?)
    }

    pub fn where_condition(condition: &Self, x: &Self, y: &Self) -> Result<Self> {
        let data = ndarray::Zip::from(&condition.data).and(&x.data).and(&y.data).map_collect(|&c, &xv, &yv| {
            if c != 0.0 { xv } else { yv }
        });
        Ok(Self {
            data,
            device: x.device,
            requires_grad: x.requires_grad || y.requires_grad,
        })
    }

    pub fn nonzero(&self) -> Result<Self> {
        let mut indices = Vec::new();
        for (idx, &val) in self.data.indexed_iter() {
            if val != 0.0 {
                for &i in idx.slice() {
                    indices.push(i as TensorDtype);
                }
            }
        }
        let nnz = indices.len() / self.ndim().max(1);
        let shape = if nnz == 0 { vec![self.ndim(), 0] } else { vec![self.ndim(), nnz] };
        Self::from_slice(&indices, &shape)
    }

    pub fn scatter_(&mut self, dim: usize, index: &Self, src: &Self) -> Result<()> {
        let index_data = &index.data;
        let src_data = &src.data;

        for (idx, &src_idx) in index_data.indexed_iter() {
            let idx_vec: Vec<usize> = idx.slice().iter().map(|&i| i).collect();
            let mut target_idx = idx_vec.clone();
            target_idx[dim] = src_idx as usize;
            if let Some(&val) = src_data.get(idx_vec.as_slice()) {
                if let Some(elem) = self.data.get_mut(target_idx.as_slice()) {
                    *elem = val;
                }
            }
        }
        Ok(())
    }

    pub fn gather_(&mut self, dim: usize, index: &Self, src: &Self) -> Result<()> {
        let src_data = &src.data;

        for (idx, &gather_idx) in index.data.indexed_iter() {
            let idx_vec: Vec<usize> = idx.slice().iter().map(|&i| i).collect();
            let mut src_idx = idx_vec.clone();
            src_idx[dim] = gather_idx as usize;
            if let Some(&val) = src_data.get(src_idx.as_slice()) {
                let target_idx: Vec<usize> = idx_vec.iter().map(|&i| i).collect();
                if let Some(elem) = self.data.get_mut(target_idx.as_slice()) {
                    *elem = val;
                }
            }
        }
        Ok(())
    }

    pub fn gather(&self, dim: usize, index: &Self) -> Result<Self> {
        let mut result_data = index.data.clone();

        for (idx, &gather_idx) in index.data.indexed_iter() {
            let idx_vec: Vec<usize> = idx.slice().iter().map(|&i| i).collect();
            let mut src_idx = idx_vec.clone();
            src_idx[dim] = gather_idx as usize;
            if let Some(&val) = self.data.get(src_idx.as_slice()) {
                if let Some(elem) = result_data.get_mut(idx_vec.as_slice()) {
                    *elem = val;
                }
            }
        }

        Ok(Self {
            data: result_data,
            device: self.device,
            requires_grad: self.requires_grad,
        })
    }

    pub fn index_add(&mut self, dim: usize, index: &Self, source: &Self) -> Result<()> {
        let index_vec: Vec<usize> = index.data.iter().map(|&x| x as usize).collect();

        for (i, &idx) in index_vec.iter().enumerate() {
            let mut src_coords = vec![i];
            for _ in 1..source.ndim() {
                src_coords.push(0);
            }
            if let Some(&val) = source.data.get(src_coords.as_slice()) {
                let mut target_coords = vec![idx];
                for _ in 1..self.ndim() {
                    target_coords.push(0);
                }
                if let Some(elem) = self.data.get_mut(target_coords.as_slice()) {
                    *elem += val;
                }
            }
        }
        Ok(())
    }

    pub fn index_copy(&mut self, dim: usize, index: &Self, source: &Self) -> Result<()> {
        self.scatter_(dim, index, source)
    }

    pub fn index_fill(&mut self, dim: usize, index: &Self, value: TensorDtype) -> Result<()> {
        let index_vec: Vec<usize> = index.data.iter().map(|&x| x as usize).collect();

        for &idx in &index_vec {
            let mut coords = vec![idx];
            for _ in 1..self.ndim() {
                coords.push(0);
            }
            if let Some(elem) = self.data.get_mut(coords.as_slice()) {
                *elem = value;
            }
        }
        Ok(())
    }

    pub fn vstack(tensors: &[&Self]) -> Result<Self> {
        if tensors.is_empty() {
            return Err(LeorchError::InvalidOperation("Cannot vstack empty list".to_string()));
        }
        let unsqueezed: Result<Vec<_>> = tensors.iter().map(|t| t.unsqueeze(0)).collect();
        Self::cat(&unsqueezed?.iter().collect::<Vec<_>>(), 0)
    }

    pub fn hstack(tensors: &[&Self]) -> Result<Self> {
        if tensors.is_empty() {
            return Err(LeorchError::InvalidOperation("Cannot hstack empty list".to_string()));
        }
        if tensors[0].ndim() == 1 {
            Self::cat(tensors, 0)
        } else {
            Self::cat(tensors, 1)
        }
    }

    pub fn dstack(tensors: &[&Self]) -> Result<Self> {
        if tensors.is_empty() {
            return Err(LeorchError::InvalidOperation("Cannot dstack empty list".to_string()));
        }
        let unsqueezed: Result<Vec<_>> = tensors.iter().map(|t| t.unsqueeze(2)).collect();
        Self::cat(&unsqueezed?.iter().collect::<Vec<_>>(), 2)
    }

    pub fn split(&self, split_size: usize, dim: usize) -> Result<Vec<Self>> {
        let size = self.shape()[dim];
        let num_splits = (size + split_size - 1) / split_size;
        let mut result = Vec::new();

        for i in 0..num_splits {
            let start = i * split_size;
            let end = (start + split_size).min(size);
            let slice_size = end - start;

            let mut new_shape = self.shape();
            new_shape[dim] = slice_size;

            let mut data = Vec::new();
            for idx in 0..self.numel() {
                let coords: Vec<usize> = self.data.raw_dim().as_array_view().iter().enumerate().map(|(d, &s)| idx / s).collect::<Vec<_>>();
                if coords[dim] >= start && coords[dim] < end {
                    data.push(self.data[coords.as_slice()]);
                }
            }

            result.push(Self::from_slice(&data, &new_shape)?);
        }
        Ok(result)
    }

    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Self>> {
        let size = self.shape()[dim];
        let split_size = (size + chunks - 1) / chunks;
        self.split(split_size, dim)
    }

    pub fn tile(&self, reps: &[usize]) -> Result<Self> {
        let self_shape = self.shape();
        if reps.len() != self_shape.len() {
            return Err(LeorchError::ShapeMismatch {
                expected: self_shape.clone(),
                got: reps.to_vec(),
            });
        }

        let mut result = self.clone();
        for (dim, &rep) in reps.iter().enumerate() {
            let to_concat: Vec<_> = (0..rep).map(|_| &result).collect();
            result = Self::cat(&to_concat, dim)?;
        }
        Ok(result)
    }

    pub fn repeat(&self, repeats: &[usize]) -> Result<Self> {
        let self_shape = self.shape();
        if repeats.len() != self_shape.len() {
            return Err(LeorchError::ShapeMismatch {
                expected: self_shape.clone(),
                got: repeats.to_vec(),
            });
        }

        let mut result_data = self.data.clone();
        for (dim, &rep) in repeats.iter().enumerate().rev() {
            let old_shape = result_data.raw_dim();
            let new_shape: Vec<usize> = old_shape.slice().iter().enumerate().map(|(d, &s)| {
                if d == dim { s * rep } else { s }
            }).collect();

            let mut new_data = Array::zeros(IxDyn(&new_shape));
            for (idx, &val) in result_data.indexed_iter() {
                for r in 0..rep {
                    let mut new_idx: Vec<usize> = idx.slice().iter().map(|&i| i).collect();
                    new_idx[dim] = idx[dim] + r * old_shape[dim];
                    new_data[new_idx.as_slice()] = val;
                }
            }
            result_data = new_data;
        }

        Ok(Self {
            data: result_data,
            device: self.device,
            requires_grad: self.requires_grad,
        })
    }

    pub fn unbind(&self, dim: usize) -> Vec<Self> {
        let size = self.shape()[dim];
        let mut result_shape = self.shape();
        result_shape.remove(dim);

        (0..size).filter_map(|i| {
            let mut data = Vec::new();
            for (idx, &val) in self.data.indexed_iter() {
                let coords: Vec<usize> = self.data.raw_dim().as_array_view().iter().enumerate().map(|(d, &s)| idx % s).collect();
                if coords[dim] == i {
                    data.push(val);
                }
            }
            Self::from_slice(&data, &result_shape).ok()
        }).collect()
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape())
            .field("device", &self.device)
            .field("requires_grad", &self.requires_grad)
            .field("data", &self.data)
            .finish()
    }
}

impl Add for Tensor {
    type Output = Result<Self>;
    fn add(self, other: Self) -> Self::Output {
        Tensor::add(&self, &other)
    }
}

impl Sub for Tensor {
    type Output = Result<Self>;
    fn sub(self, other: Self) -> Self::Output {
        Tensor::sub(&self, &other)
    }
}

impl Mul for Tensor {
    type Output = Result<Self>;
    fn mul(self, other: Self) -> Self::Output {
        Tensor::mul(&self, &other)
    }
}

impl Div for Tensor {
    type Output = Result<Self>;
    fn div(self, other: Self) -> Self::Output {
        Tensor::div(&self, &other)
    }
}

impl Neg for Tensor {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Tensor::neg(&self)
    }
}
