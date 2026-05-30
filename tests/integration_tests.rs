//! Integration tests for Leorch

use leorch::tensor::Tensor;
use leorch::nn::{Module, Linear, ReLU, Sigmoid, Conv2d, MaxPool2d, Flatten};
use leorch::loss::{Loss, MSELoss, CrossEntropyLoss, BCELoss};
use leorch::data::{xor_dataset, TensorDataset, DataLoader, Dataset};

#[test]
fn test_tensor_creation_and_operations() {
    // Test zeros and ones
    let zeros = Tensor::zeros(&[2, 3]);
    assert_eq!(zeros.shape(), vec![2, 3]);
    assert!(zeros.data().iter().all(|&x| x == 0.0));

    let ones = Tensor::ones(&[2, 3]);
    assert!(ones.data().iter().all(|&x| x == 1.0));

    // Test from_slice
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_slice(&data, &[2, 2]).unwrap();
    assert_eq!(tensor.shape(), vec![2, 2]);
    assert_eq!(tensor.get(&[0, 0]), Some(1.0));
    assert_eq!(tensor.get(&[1, 1]), Some(4.0));
}

#[test]
fn test_tensor_arithmetic() {
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2]).unwrap();

    // Addition
    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec(), vec![2.0, 3.0, 4.0, 5.0]);

    // Subtraction
    let d = a.sub(&b).unwrap();
    assert_eq!(d.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);

    // Element-wise multiplication
    let e = a.mul(&b).unwrap();
    assert_eq!(e.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_tensor_reshape() {
    let tensor = Tensor::ones(&[2, 3, 4]);
    assert_eq!(tensor.numel(), 24);

    let reshaped = tensor.reshape(&[6, 4]).unwrap();
    assert_eq!(reshaped.shape(), vec![6, 4]);

    let flattened = tensor.flatten();
    assert_eq!(flattened.shape(), vec![24]);
}

#[test]
fn test_tensor_matmul() {
    let a = Tensor::ones(&[2, 3]);
    let b = Tensor::ones(&[3, 4]);
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.shape(), vec![2, 4]);
    // Each element should be sum of 3 ones = 3
    assert!(c.data().iter().all(|&x| x == 3.0));
}

#[test]
fn test_linear_layer() {
    let linear = Linear::new(10, 5);
    let input = Tensor::ones(&[2, 10]);
    let output = linear.forward(&input);

    assert_eq!(output.shape(), vec![2, 5]);

    // Check parameters
    let params = linear.parameters();
    assert_eq!(params.len(), 2); // weight and bias
}

#[test]
fn test_activation_functions() {
    let input = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();

    // ReLU
    let relu = ReLU::new();
    let relu_out = relu.forward(&input);
    assert_eq!(relu_out.to_vec(), vec![0.0, 0.0, 1.0, 2.0]);

    // Sigmoid
    let sigmoid = Sigmoid::new();
    let sigmoid_out = sigmoid.forward(&input);
    // sigmoid(0) = 0.5
    assert!((sigmoid_out.get(&[1]).unwrap() - 0.5).abs() < 1e-6);

    // All sigmoid outputs should be in (0, 1)
    for val in sigmoid_out.to_vec() {
        assert!(val > 0.0 && val < 1.0);
    }
}

#[test]
fn test_conv2d_layer() {
    let conv = Conv2d::new(3, 16, (3, 3));
    let input = Tensor::ones(&[1, 3, 32, 32]);
    let output = conv.forward(&input);

    // Output size: (32 - 3 + 1) = 30
    assert_eq!(output.shape(), vec![1, 16, 30, 30]);

    // Check parameters
    let params = conv.parameters();
    assert_eq!(params.len(), 2); // weight and bias
}

#[test]
fn test_max_pool2d() {
    let pool = MaxPool2d::new((2, 2));
    let input = Tensor::ones(&[1, 3, 32, 32]);
    let output = pool.forward(&input);

    assert_eq!(output.shape(), vec![1, 3, 16, 16]);
}

#[test]
fn test_flatten() {
    let flatten = Flatten::new();
    let input = Tensor::ones(&[2, 3, 4, 5]);
    let output = flatten.forward(&input);

    assert_eq!(output.shape(), vec![2, 60]);
}

#[test]
fn test_mse_loss() {
    let prediction = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let target = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let loss = MSELoss::new();
    let result = loss.forward(&prediction, &target);

    // MSE of identical tensors should be 0
    assert!(result.to_vec()[0].abs() < 1e-6);

    // Test with different values
    let target2 = Tensor::from_slice(&[2.0, 3.0, 4.0], &[3]).unwrap();
    let result2 = loss.forward(&prediction, &target2);
    // MSE = mean((1-2)^2, (2-3)^2, (3-4)^2) = mean(1, 1, 1) = 1
    assert!((result2.to_vec()[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_bce_loss() {
    // Perfect prediction
    let prediction = Tensor::from_slice(&[0.9, 0.1, 0.8], &[3]).unwrap();
    let target = Tensor::from_slice(&[1.0, 0.0, 1.0], &[3]).unwrap();
    let loss = BCELoss::new();
    let result = loss.forward(&prediction, &target);

    // Loss should be small for good predictions
    assert!(result.to_vec()[0] < 0.5);
}

#[test]
fn test_cross_entropy_loss() {
    // Perfect prediction (high logit for correct class)
    let prediction = Tensor::from_slice(&[10.0, 0.0, 0.0], &[1, 3]).unwrap();
    let target = Tensor::from_slice(&[0.0], &[1]).unwrap(); // Class 0
    let loss = CrossEntropyLoss::new();
    let result = loss.forward(&prediction, &target);

    // Loss should be small for good predictions
    assert!(result.to_vec()[0] < 0.1);
}

#[test]
fn test_xor_dataset() {
    let dataset = xor_dataset();
    assert_eq!(dataset.len(), 4);

    // Check XOR truth table
    let (sample, target) = dataset.get(0).unwrap();
    assert_eq!(sample.to_vec(), vec![0.0, 0.0]);
    assert_eq!(target.to_vec(), vec![0.0]);

    let (sample, target) = dataset.get(1).unwrap();
    assert_eq!(sample.to_vec(), vec![0.0, 1.0]);
    assert_eq!(target.to_vec(), vec![1.0]);
}

#[test]
fn test_data_loader() {
    let data = Tensor::zeros(&[20, 5]);
    let targets = Tensor::zeros(&[20]);
    let dataset = TensorDataset::new(data, targets).unwrap();
    let mut loader = DataLoader::new(dataset, 4);

    let mut batch_count = 0;
    for (batch_data, batch_targets) in &mut loader {
        assert_eq!(batch_data.shape()[0], 4);
        assert_eq!(batch_targets.shape()[0], 4);
        batch_count += 1;
    }

    assert_eq!(batch_count, 5);
}

#[test]
fn test_tensor_serialization() {
    use serde::{Serialize, Deserialize};

    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    // Serialize
    let serialized = serde_json::to_string(&tensor).expect("Failed to serialize tensor");

    // Deserialize
    let deserialized: Tensor = serde_json::from_str(&serialized).expect("Failed to deserialize tensor");

    assert_eq!(tensor.shape(), deserialized.shape());
    assert_eq!(tensor.to_vec(), deserialized.to_vec());
}

#[test]
fn test_tensor_reductions() {
    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    // Sum
    let sum = tensor.sum();
    assert!((sum - 21.0).abs() < 1e-6);

    // Mean
    let mean = tensor.mean();
    assert!((mean - 3.5).abs() < 1e-6);

    // Max
    let max = tensor.max();
    assert!((max - 6.0).abs() < 1e-6);

    // Min
    let min = tensor.min();
    assert!((min - 1.0).abs() < 1e-6);
}

#[test]
fn test_tensor_unsqueeze_squeeze() {
    let tensor = Tensor::ones(&[3, 4]);

    // Unsqueeze
    let unsqueezed = tensor.unsqueeze(0).unwrap();
    assert_eq!(unsqueezed.shape(), vec![1, 3, 4]);

    let unsqueezed = tensor.unsqueeze(2).unwrap();
    assert_eq!(unsqueezed.shape(), vec![3, 4, 1]);

    // Squeeze
    let squeezed = unsqueezed.squeeze(Some(2));
    assert_eq!(squeezed.shape(), vec![3, 4]);
}

#[test]
fn test_tensor_transpose() {
    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let transposed = tensor.transpose(0, 1).unwrap();

    assert_eq!(transposed.shape(), vec![3, 2]);
    assert_eq!(transposed.get(&[0, 0]), Some(1.0));
    assert_eq!(transposed.get(&[0, 1]), Some(4.0));
    assert_eq!(transposed.get(&[1, 0]), Some(2.0));
    assert_eq!(transposed.get(&[1, 1]), Some(5.0));
}

#[test]
fn test_tensor_permute() {
    let tensor = Tensor::ones(&[2, 3, 4]);
    let permuted = tensor.permute(&[2, 0, 1]).unwrap();

    assert_eq!(permuted.shape(), vec![4, 2, 3]);
}

#[test]
fn test_tensor_element_wise_ops() {
    let tensor = Tensor::from_slice(&[1.0, 4.0, 9.0], &[3]).unwrap();

    // Sqrt
    let sqrt = tensor.sqrt();
    assert_eq!(sqrt.to_vec(), vec![1.0, 2.0, 3.0]);

    // Power
    let squared = tensor.pow(2.0);
    assert_eq!(squared.to_vec(), vec![1.0, 16.0, 81.0]);

    // Abs
    let negative = Tensor::from_slice(&[-1.0, -2.0, 3.0], &[3]).unwrap();
    let abs = negative.abs();
    assert_eq!(abs.to_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_linear_without_bias() {
    let linear = Linear::new_without_bias(5, 3);
    let input = Tensor::ones(&[2, 5]);
    let output = linear.forward(&input);

    assert_eq!(output.shape(), vec![2, 3]);

    // Should only have weight parameter
    let params = linear.parameters();
    assert_eq!(params.len(), 1);
}

#[test]
fn test_conv2d_with_padding() {
    let conv = Conv2d::new_with_params(3, 16, (3, 3), (1, 1), (1, 1), (1, 1), true);
    let input = Tensor::ones(&[1, 3, 32, 32]);
    let output = conv.forward(&input);

    // With padding=1, output size should be same as input
    assert_eq!(output.shape(), vec![1, 16, 32, 32]);
}

#[test]
fn test_loss_reductions() {
    use leorch::loss::{MSELoss, Reduction};

    let prediction = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let target = Tensor::from_slice(&[2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();

    // Mean reduction (default)
    let loss_mean = MSELoss::with_reduction(Reduction::Mean);
    let result_mean = loss_mean.forward(&prediction, &target);
    // MSE = mean((1-2)^2, (2-3)^2, (3-4)^2, (4-5)^2) = mean(1, 1, 1, 1) = 1
    assert!((result_mean.to_vec()[0] - 1.0).abs() < 1e-6);

    // Sum reduction
    let loss_sum = MSELoss::with_reduction(Reduction::Sum);
    let result_sum = loss_sum.forward(&prediction, &target);
    // MSE = sum(1, 1, 1, 1) = 4
    assert!((result_sum.to_vec()[0] - 4.0).abs() < 1e-6);
}
