use crate::tensor::{Tensor, TensorDtype};
use crate::error::Result;

pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<Tensor> {
    let input_shape = input.shape();
    let weight_shape = weight.shape();

    if input_shape.len() != 4 {
        return Err(crate::error::LeorchError::DimensionError(
            format!("Expected 4D input [N, C, H, W], got {:?}", input_shape)
        ));
    }

    if weight_shape.len() != 4 {
        return Err(crate::error::LeorchError::DimensionError(
            format!("Expected 4D weight [C_out, C_in, K_h, K_w], got {:?}", weight_shape)
        ));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let out_channels = weight_shape[0];
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];

    let out_height = ((in_height + 2 * padding.0 - dilation.0 * (kernel_h - 1) - 1) / stride.0) + 1;
    let out_width = ((in_width + 2 * padding.1 - dilation.1 * (kernel_w - 1) - 1) / stride.1) + 1;

    let mut output = Tensor::zeros(&[batch_size, out_channels, out_height, out_width]);
    let input_data = input.data();
    let weight_data = weight.data();
    let output_data = output.data_mut();

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

    if let Some(bias) = bias {
        let bias_data = bias.data();
        for n in 0..batch_size {
            for oc in 0..out_channels {
                let bias_val = bias_data[[oc]];
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        output_data[[n, oc, oh, ow]] += bias_val;
                    }
                }
            }
        }
    }

    Ok(output)
}

pub fn max_pool2d(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<Tensor> {
    let stride = stride.unwrap_or(kernel_size);
    let input_shape = input.shape();

    if input_shape.len() != 4 {
        return Err(crate::error::LeorchError::DimensionError(
            format!("Expected 4D input [N, C, H, W], got {:?}", input_shape)
        ));
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let out_height = ((in_height + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1) / stride.0) + 1;
    let out_width = ((in_width + 2 * padding.1 - dilation.1 * (kernel_size.1 - 1) - 1) / stride.1) + 1;

    let mut output = Tensor::zeros(&[batch_size, channels, out_height, out_width]);
    let input_data = input.data();
    let output_data = output.data_mut();

    for n in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut max_val = TensorDtype::NEG_INFINITY;

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let ih = oh * stride.0 + kh * dilation.0;
                            let iw = ow * stride.1 + kw * dilation.1;

                            let ih_padded = ih as i64 - padding.0 as i64;
                            let iw_padded = iw as i64 - padding.1 as i64;

                            if ih_padded >= 0 && ih_padded < in_height as i64 &&
                               iw_padded >= 0 && iw_padded < in_width as i64 {
                                let val = input_data[[n, c, ih_padded as usize, iw_padded as usize]];
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

    Ok(output)
}

pub fn avg_pool2d(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    count_include_pad: bool,
) -> Result<Tensor> {
    let stride = stride.unwrap_or(kernel_size);
    let input_shape = input.shape();

    if input_shape.len() != 4 {
        return Err(crate::error::LeorchError::DimensionError(
            format!("Expected 4D input [N, C, H, W], got {:?}", input_shape)
        ));
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let out_height = ((in_height + 2 * padding.0 - kernel_size.0) / stride.0) + 1;
    let out_width = ((in_width + 2 * padding.1 - kernel_size.1) / stride.1) + 1;

    let mut output = Tensor::zeros(&[batch_size, channels, out_height, out_width]);
    let input_data = input.data();
    let output_data = output.data_mut();

    let kernel_area = (kernel_size.0 * kernel_size.1) as TensorDtype;

    for n in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut sum = 0.0;
                    let mut count = 0;

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let ih = oh * stride.0 + kh;
                            let iw = ow * stride.1 + kw;

                            let ih_padded = ih as i64 - padding.0 as i64;
                            let iw_padded = iw as i64 - padding.1 as i64;

                            if ih_padded >= 0 && ih_padded < in_height as i64 &&
                               iw_padded >= 0 && iw_padded < in_width as i64 {
                                sum += input_data[[n, c, ih_padded as usize, iw_padded as usize]];
                                count += 1;
                            }
                        }
                    }

                    let divisor = if count_include_pad {
                        kernel_area
                    } else {
                        count as TensorDtype
                    };

                    output_data[[n, c, oh, ow]] = sum / divisor;
                }
            }
        }
    }

    Ok(output)
}

pub fn adaptive_avg_pool2d(input: &Tensor, output_size: (usize, usize)) -> Result<Tensor> {
    let input_shape = input.shape();

    if input_shape.len() != 4 {
        return Err(crate::error::LeorchError::DimensionError(
            format!("Expected 4D input [N, C, H, W], got {:?}", input_shape)
        ));
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let mut output = Tensor::zeros(&[batch_size, channels, output_size.0, output_size.1]);
    let input_data = input.data();
    let output_data = output.data_mut();

    for n in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_size.0 {
                for ow in 0..output_size.1 {
                    let h_start = (oh * in_height) / output_size.0;
                    let h_end = ((oh + 1) * in_height) / output_size.0;
                    let w_start = (ow * in_width) / output_size.1;
                    let w_end = ((ow + 1) * in_width) / output_size.1;

                    let mut sum = 0.0;
                    let count = ((h_end - h_start) * (w_end - w_start)) as TensorDtype;

                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            sum += input_data[[n, c, ih, iw]];
                        }
                    }

                    output_data[[n, c, oh, ow]] = sum / count;
                }
            }
        }
    }

    Ok(output)
}

pub fn pad(input: &Tensor, padding: &[usize], value: TensorDtype) -> Result<Tensor> {
    let input_shape = input.shape();
    let ndim = input_shape.len();

    if padding.len() % 2 != 0 || padding.len() / 2 > ndim {
        return Err(crate::error::LeorchError::DimensionError(
            "Invalid padding specification".to_string()
        ));
    }

    let pad_dims = padding.len() / 2;
    let mut output_shape = input_shape.clone();

    for i in 0..pad_dims {
        let pad_left = padding[2 * i];
        let pad_right = padding[2 * i + 1];
        output_shape[ndim - 1 - i] += pad_left + pad_right;
    }

    let mut output = Tensor::zeros(&output_shape);
    output = output.add_scalar(value);

    Ok(output)
}

pub fn relu(input: &Tensor) -> Tensor {
    Tensor {
        data: input.data().mapv(|x| x.max(0.0)),
        device: input.device,
        requires_grad: input.requires_grad,
    }
}

pub fn sigmoid(input: &Tensor) -> Tensor {
    Tensor {
        data: input.data().mapv(|x| 1.0 / (1.0 + (-x).exp())),
        device: input.device,
        requires_grad: input.requires_grad,
    }
}

pub fn tanh(input: &Tensor) -> Tensor {
    input.tanh()
}

pub fn gelu(input: &Tensor) -> Tensor {
    const SQRT_2_OVER_PI: TensorDtype = 0.7978845608;
    const COEFF: TensorDtype = 0.044715;

    let x_cubed = input.pow(3.0);
    let inner = input.add(&x_cubed.mul_scalar(COEFF)).unwrap();
    let tanh_arg = inner.mul_scalar(SQRT_2_OVER_PI);
    let tanh_result = tanh_arg.tanh();
    let half = tanh_result.add_scalar(1.0);
    half.mul_scalar(0.5).mul(input).unwrap()
}

pub fn softmax(input: &Tensor, dim: usize) -> Result<Tensor> {
    let max_val = input.max();
    let shifted = input.add_scalar(-max_val);
    let exp = shifted.exp();
    let sum = exp.sum_dim(dim, true)?;
    exp.div(&sum)
}

pub fn log_softmax(input: &Tensor, dim: usize) -> Result<Tensor> {
    let max_val = input.max();
    let shifted = input.add_scalar(-max_val);
    let exp = shifted.exp();
    let sum = exp.sum_dim(dim, true)?;
    let log_sum = sum.log();
    shifted.sub(&log_sum)
}

pub fn dropout(input: &Tensor, p: TensorDtype, training: bool) -> Tensor {
    if !training || p == 0.0 {
        return input.clone();
    }

    if p == 1.0 {
        return Tensor::zeros(&input.shape());
    }

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let scale = 1.0 / (1.0 - p);

    let mask_data: Vec<TensorDtype> = input.data()
        .iter()
        .map(|_| if rng.gen::<TensorDtype>() > p { scale } else { 0.0 })
        .collect();

    let mask = Tensor::from_slice(&mask_data, &input.shape()).unwrap();
    input.mul(&mask).unwrap()
}

pub fn batch_norm(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    training: bool,
    momentum: TensorDtype,
    eps: TensorDtype,
) -> Result<Tensor> {
    let shape = input.shape();
    let num_features = shape[1];

    if training {
        let mean = input.mean_dim(1, true)?;
        let var = input.pow(2.0).mean_dim(1, true)?;

        let normalized = input.sub(&mean)?;
        let normalized = normalized.div(&var.sqrt().add_scalar(eps))?;

        let mut output = normalized;
        if let Some(weight) = weight {
            let weight_reshaped = weight.reshape(&[1, num_features, 1, 1])?;
            output = output.mul(&weight_reshaped)?;
        }
        if let Some(bias) = bias {
            let bias_reshaped = bias.reshape(&[1, num_features, 1, 1])?;
            output = output.add(&bias_reshaped)?;
        }

        Ok(output)
    } else {
        let mean = running_mean.ok_or_else(|| {
            crate::error::LeorchError::InvalidOperation(
                "Running mean required for eval mode".to_string()
            )
        })?;
        let var = running_var.ok_or_else(|| {
            crate::error::LeorchError::InvalidOperation(
                "Running var required for eval mode".to_string()
            )
        })?;

        let mean_reshaped = mean.reshape(&[1, num_features, 1, 1])?;
        let var_reshaped = var.reshape(&[1, num_features, 1, 1])?;

        let normalized = input.sub(&mean_reshaped)?;
        let normalized = normalized.div(&var_reshaped.sqrt().add_scalar(eps))?;

        let mut output = normalized;
        if let Some(weight) = weight {
            let weight_reshaped = weight.reshape(&[1, num_features, 1, 1])?;
            output = output.mul(&weight_reshaped)?;
        }
        if let Some(bias) = bias {
            let bias_reshaped = bias.reshape(&[1, num_features, 1, 1])?;
            output = output.add(&bias_reshaped)?;
        }

        Ok(output)
    }
}

pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let output = input.matmul(&weight.transpose(0, 1)?)?;

    if let Some(bias) = bias {
        let bias_expanded = bias.unsqueeze(0)?;
        output.add(&bias_expanded)
    } else {
        Ok(output)
    }
}

pub fn tensordot(a: &Tensor, b: &Tensor, dims_a: &[usize], dims_b: &[usize]) -> Result<Tensor> {
if dims_a.len() != dims_b.len() {
return Err(crate::error::LeorchError::DimensionError(
"Number of contraction dimensions must match".to_string()
));
}

let a_shape = a.shape();
let b_shape = b.shape();

for (i, (&da, &db)) in dims_a.iter().zip(dims_b.iter()).enumerate() {
if da >= a_shape.len() || db >= b_shape.len() {
return Err(crate::error::LeorchError::DimensionError(
format!("Contraction dimension out of bounds at index {}", i)
));
}
if a_shape[da] != b_shape[db] {
return Err(crate::error::LeorchError::ShapeMismatch {
expected: vec![a_shape[da]],
got: vec![b_shape[db]],
});
}
}

let mut a_perm: Vec<usize> = (0..a_shape.len()).filter(|i| !dims_a.contains(i)).collect();
a_perm.extend(dims_a);

let mut b_perm: Vec<usize> = dims_b.to_vec();
b_perm.extend((0..b_shape.len()).filter(|i| !dims_b.contains(i)));

let a_permuted = a.permute(&a_perm)?;
let b_permuted = b.permute(&b_perm)?;

let a_new_shape: Vec<usize> = a_perm.iter().map(|&i| a_shape[i]).collect();
let b_new_shape: Vec<usize> = b_perm.iter().map(|&i| b_shape[i]).collect();

let a_contraction_size: usize = dims_a.iter().map(|&i| a_shape[i]).product();
let a_remaining_size: usize = a_new_shape.iter().take(a_new_shape.len() - dims_a.len()).product();
let b_remaining_size: usize = b_new_shape.iter().skip(dims_b.len()).product();

let a_reshaped = a_permuted.reshape(&[a_remaining_size, a_contraction_size])?;
let b_reshaped = b_permuted.reshape(&[a_contraction_size, b_remaining_size])?;

let result = a_reshaped.matmul(&b_reshaped)?;

let output_shape: Vec<usize> = a_new_shape.iter().take(a_new_shape.len() - dims_a.len())
.chain(b_new_shape.iter().skip(dims_b.len()))
.copied()
.collect();

result.reshape(&output_shape)
}

pub fn kron(a: &Tensor, b: &Tensor) -> Result<Tensor> {
let a_shape = a.shape();
let b_shape = b.shape();

if a_shape.len() != b_shape.len() {
return Err(crate::error::LeorchError::DimensionError(
"Kronecker product requires same number of dimensions".to_string()
));
}

let output_shape: Vec<usize> = a_shape.iter().zip(b_shape.iter())
.map(|(&as_, &bs)| as_ * bs)
.collect();

let mut output = Tensor::zeros(&output_shape);
let output_data = output.data_mut();
let a_data = a.data();
let b_data = b.data();

let ndim = a_shape.len();
let a_size: usize = a_shape.iter().product();
let b_size: usize = b_shape.iter().product();

for a_idx in 0..a_size {
let mut a_coords = vec![0; ndim];
let mut temp = a_idx;
for i in (0..ndim).rev() {
a_coords[i] = temp % a_shape[i];
temp /= a_shape[i];
}

for b_idx in 0..b_size {
let mut b_coords = vec![0; ndim];
let mut temp = b_idx;
for i in (0..ndim).rev() {
b_coords[i] = temp % b_shape[i];
temp /= b_shape[i];
}

let mut out_coords = vec![0; ndim];
for i in 0..ndim {
out_coords[i] = a_coords[i] * b_shape[i] + b_coords[i];
}

let a_val = a_data[&a_coords[..]];
let b_val = b_data[&b_coords[..]];
output_data[&out_coords[..]] = a_val * b_val;
}
}

Ok(output)
}

pub fn einsum(equation: &str, tensors: &[&Tensor]) -> Result<Tensor> {
let parts: Vec<&str> = equation.split("->").collect();
if parts.len() != 2 {
return Err(crate::error::LeorchError::InvalidOperation(
"Einsum equation must contain '->'".to_string()
));
}

let input_part = parts[0].trim();
let output_part = parts[1].trim();

let input_specs: Vec<&str> = input_part.split(',').map(|s| s.trim()).collect();
if input_specs.len() != tensors.len() {
return Err(crate::error::LeorchError::DimensionError(
format!("Expected {} input tensors, got {}", input_specs.len(), tensors.len())
));
}

let mut char_to_size: std::collections::HashMap<char, usize> = std::collections::HashMap::new();

for (i, (spec, tensor)) in input_specs.iter().zip(tensors.iter()).enumerate() {
let shape = tensor.shape();
if spec.len() != shape.len() {
return Err(crate::error::LeorchError::DimensionError(
format!("Tensor {} has {} dimensions but spec has {}", i, shape.len(), spec.len())
));
}
for (j, c) in spec.chars().enumerate() {
let size = shape[j];
if let Some(&existing) = char_to_size.get(&c) {
if existing != size {
return Err(crate::error::LeorchError::ShapeMismatch {
expected: vec![existing],
got: vec![size],
});
}
} else {
char_to_size.insert(c, size);
}
}
}

let output_shape: Vec<usize> = output_part.chars()
.filter(|c| c.is_alphabetic())
.map(|c| *char_to_size.get(&c).unwrap_or(&1))
.collect();

if output_part.len() == 1 && output_part.chars().next() == Some('.') {
return Ok(Tensor::zeros(&[1]));
}

if output_part.chars().all(|c| !c.is_alphabetic()) {
let mut result = 0.0;
for tensor in tensors {
result += tensor.sum();
}
return Ok(Tensor::from_slice(&[result], &[1]).unwrap());
}

Ok(Tensor::zeros(&output_shape))
}