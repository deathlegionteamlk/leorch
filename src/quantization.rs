use crate::tensor::{Tensor, TensorDtype};
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationScheme {
    PerTensorAffine,
    PerChannelAffine,
    PerTensorSymmetric,
    PerChannelSymmetric,
}

#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    data: Vec<i8>,
    scale: TensorDtype,
    zero_point: i32,
    shape: Vec<usize>,
    scheme: QuantizationScheme,
}

impl QuantizedTensor {
    pub fn from_tensor(
        tensor: &Tensor,
        scale: TensorDtype,
        zero_point: i32,
        scheme: QuantizationScheme,
    ) -> Self {
        let shape = tensor.shape();
        let data: Vec<i8> = tensor.data()
            .iter()
            .map(|&x| {
                let quantized = (x / scale).round() as i32 + zero_point;
                quantized.clamp(-128, 127) as i8
            })
            .collect();
        
        Self {
            data,
            scale,
            zero_point,
            shape,
            scheme,
        }
    }
    
    pub fn to_tensor(&self) -> Tensor {
        let dequantized: Vec<TensorDtype> = self.data
            .iter()
            .map(|&x| {
                (x as i32 - self.zero_point) as TensorDtype * self.scale
            })
            .collect();
        
        Tensor::from_slice(&dequantized, &self.shape).unwrap_or_else(|_| Tensor::zeros(&[1]))
    }
    
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

pub fn quantize_per_tensor(
    tensor: &Tensor,
    scale: TensorDtype,
    zero_point: i32,
) -> QuantizedTensor {
    QuantizedTensor::from_tensor(tensor, scale, zero_point, QuantizationScheme::PerTensorAffine)
}

pub fn quantize_per_channel(
    tensor: &Tensor,
    scales: &[TensorDtype],
    zero_points: &[i32],
    axis: usize,
) -> QuantizedTensor {
    QuantizedTensor::from_tensor(
        tensor,
        scales[0],
        zero_points[0],
        QuantizationScheme::PerChannelAffine,
    )
}

pub fn calculate_qparams(
    tensor: &Tensor,
    num_bits: usize,
) -> (TensorDtype, i32) {
    let min_val = tensor.data().iter().fold(TensorDtype::INFINITY, |a, &b| a.min(b));
    let max_val = tensor.data().iter().fold(TensorDtype::NEG_INFINITY, |a, &b| a.max(b));
    
    let qmin = -(1i32 << (num_bits - 1));
    let qmax = (1i32 << (num_bits - 1)) - 1;
    
    let scale = (max_val - min_val) / ((qmax - qmin) as TensorDtype);
    let zero_point = ((-min_val / scale).round() as i32 + qmin).clamp(qmin, qmax);
    
    (scale, zero_point)
}

pub struct QuantizedLinear {
    weight: QuantizedTensor,
    bias: Option<Tensor>,
    input_scale: TensorDtype,
    input_zero_point: i32,
    output_scale: TensorDtype,
    output_zero_point: i32,
}

impl QuantizedLinear {
    pub fn new(
        weight: QuantizedTensor,
        bias: Option<Tensor>,
        input_scale: TensorDtype,
        input_zero_point: i32,
        output_scale: TensorDtype,
        output_zero_point: i32,
    ) -> Self {
        Self {
            weight,
            bias,
            input_scale,
            input_zero_point,
            output_scale,
            output_zero_point,
        }
    }
    
    pub fn forward(&self, input: &QuantizedTensor) -> QuantizedTensor {
        let input_tensor = input.to_tensor();
        let weight_tensor = self.weight.to_tensor();
        
        let output = input_tensor.matmul(&weight_tensor.transpose(0, 1).unwrap()).unwrap();
        
        let output_with_bias = if let Some(bias) = &self.bias {
            output.add(bias).unwrap()
        } else {
            output
        };
        
        QuantizedTensor::from_tensor(
            &output_with_bias,
            self.output_scale,
            self.output_zero_point,
            QuantizationScheme::PerTensorAffine,
        )
    }
}

pub struct QuantizedConv2d {
    weight: QuantizedTensor,
    bias: Option<Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    input_scale: TensorDtype,
    input_zero_point: i32,
    output_scale: TensorDtype,
    output_zero_point: i32,
}

impl QuantizedConv2d {
    pub fn new(
        weight: QuantizedTensor,
        bias: Option<Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        input_scale: TensorDtype,
        input_zero_point: i32,
        output_scale: TensorDtype,
        output_zero_point: i32,
    ) -> Self {
        Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            input_scale,
            input_zero_point,
            output_scale,
            output_zero_point,
        }
    }
}

pub fn fuse_conv_bn(
    conv_weight: &Tensor,
    conv_bias: Option<&Tensor>,
    bn_weight: &Tensor,
    bn_bias: &Tensor,
    bn_running_mean: &Tensor,
    bn_running_var: &Tensor,
    bn_eps: TensorDtype,
) -> (Tensor, Tensor) {
    let std = bn_running_var.add_scalar(bn_eps).sqrt();
    let scale = bn_weight.div(&std).unwrap();
    
    let fused_weight = conv_weight.mul(&scale.unsqueeze(1).unwrap()).unwrap();
    
    let fused_bias = if let Some(bias) = conv_bias {
        let bn_mean_scaled = bn_running_mean.mul(&scale).unwrap();
        bias.sub(&bn_mean_scaled).unwrap().add(bn_bias).unwrap()
    } else {
        bn_bias.sub(&bn_running_mean.mul(&scale).unwrap()).unwrap()
    };
    
    (fused_weight, fused_bias)
}

pub fn fuse_linear_bn(
    linear_weight: &Tensor,
    linear_bias: Option<&Tensor>,
    bn_weight: &Tensor,
    bn_bias: &Tensor,
    bn_running_mean: &Tensor,
    bn_running_var: &Tensor,
    bn_eps: TensorDtype,
) -> (Tensor, Tensor) {
    fuse_conv_bn(linear_weight, linear_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var, bn_eps)
}
