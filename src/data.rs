use crate::tensor::{Tensor, TensorDtype};
use crate::error::Result;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub trait Dataset {
    fn get(&self, index: usize) -> Option<(Tensor, Tensor)>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone)]
pub struct TensorDataset {
    data: Tensor,
    targets: Tensor,
}

impl TensorDataset {
    pub fn new(data: Tensor, targets: Tensor) -> Result<Self> {
        if data.shape()[0] != targets.shape()[0] {
            return Err(crate::error::LeorchError::ShapeMismatch {
                expected: vec![data.shape()[0]],
                got: vec![targets.shape()[0]],
            });
        }
        Ok(Self { data, targets })
    }
    
    pub fn data(&self) -> &Tensor {
        &self.data
    }
    
    pub fn targets(&self) -> &Tensor {
        &self.targets
    }
}

impl Dataset for TensorDataset {
    fn get(&self, index: usize) -> Option<(Tensor, Tensor)> {
        if index >= self.len() {
            return None;
        }

        let data_shape: Vec<usize> = self.data.shape()[1..].to_vec();
        let mut sample_shape = vec![1];
        sample_shape.extend(&data_shape);

        let data_slice = self.data.slice(index);
        let data_sample = data_slice.reshape(&sample_shape).ok()?;
            
        let target_sample = if self.targets.ndim() == 1 {
            Tensor::from_slice(&[self.targets.get(&[index]).unwrap_or(0.0)], &[1]).ok()?
            } else {
            let target_shape: Vec<usize> = self.targets.shape()[1..].to_vec();
            let mut sample_target_shape = vec![1];
        sample_target_shape.extend(&target_shape);
let target_slice = self.targets.slice(index);
        target_slice.reshape(&sample_target_shape).ok()?
    };
    
    Some((data_sample, target_sample))
    }

    fn len(&self) -> usize {
        self.data.shape()[0]
    }
}

pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    indices: Vec<usize>,
    current_index: usize,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self {
        Self::with_options(dataset, batch_size, false, false)
    }
    
    pub fn with_options(
        dataset: D,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
    ) -> Self {
        let len = dataset.len();
        let mut indices: Vec<usize> = (0..len).collect();

        if shuffle {
            indices.shuffle(&mut thread_rng());
        }

        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
            indices,
            current_index: 0,
        }
    }
    
    pub fn shuffle(&mut self) {
        self.indices.shuffle(&mut thread_rng());
        self.current_index = 0;
    }
    
    pub fn reset(&mut self) {
        self.current_index = 0;
        if self.shuffle {
            self.indices.shuffle(&mut thread_rng());
        }
    }
    
    pub fn num_batches(&self) -> usize {
        let num_samples = self.dataset.len();
        if self.drop_last {
            num_samples / self.batch_size
        } else {
            (num_samples + self.batch_size - 1) / self.batch_size
        }
    }
}

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.indices.len() {
            return None;
        }

        let remaining = self.indices.len() - self.current_index;
        let batch_size = if remaining < self.batch_size {
            if self.drop_last {
                return None;
            }
            remaining
        } else {
            self.batch_size
        };

        let batch_indices: Vec<usize> = self.indices[self.current_index..self.current_index + batch_size]
            .to_vec();
        self.current_index += batch_size;

        let mut data_samples = Vec::new();
        let mut target_samples = Vec::new();

        for idx in batch_indices {
            if let Some((data, target)) = self.dataset.get(idx) {
                data_samples.push(data);
                target_samples.push(target);
            }
        }

        if data_samples.is_empty() {
            return None;
        }

        let data_batch = stack_tensors(&data_samples)?;
        let target_batch = stack_tensors(&target_samples)?;

        Some((data_batch, target_batch))
    }
}

fn stack_tensors(tensors: &[Tensor]) -> Option<Tensor> {
    if tensors.is_empty() {
        return None;
    }

    let first_shape = tensors[0].shape();
    let total_batch: usize = tensors.len();

    let mut batch_shape = vec![total_batch];
    batch_shape.extend(&first_shape[1..]);

    let mut all_data: Vec<TensorDtype> = Vec::new();
    for tensor in tensors {
        all_data.extend(tensor.to_vec());
    }

    Tensor::from_slice(&all_data, &batch_shape).ok()
}

pub trait Transform {
    fn apply(&self, input: &Tensor) -> Tensor;
}

#[derive(Clone)]
pub struct Normalize {
    mean: TensorDtype,
    std: TensorDtype,
}

impl Normalize {
    pub fn new(mean: TensorDtype, std: TensorDtype) -> Self {
        Self { mean, std }
    }
}

impl Transform for Normalize {
    fn apply(&self, input: &Tensor) -> Tensor {
        let normalized = input.add_scalar(-self.mean);
        normalized.mul_scalar(1.0 / self.std)
    }
}

pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }
}

impl Transform for Compose {
    fn apply(&self, input: &Tensor) -> Tensor {
        let mut output = input.clone();
        for transform in &self.transforms {
            output = transform.apply(&output);
        }
        output
    }
}

#[derive(Clone)]
pub struct RandomHorizontalFlip {
    p: f64,
}

impl RandomHorizontalFlip {
    pub fn new(p: f64) -> Self {
        Self { p }
    }
}

impl Transform for RandomHorizontalFlip {
    fn apply(&self, input: &Tensor) -> Tensor {
        use rand::Rng;
        let mut rng = thread_rng();

        if rng.gen::<f64>() < self.p {
            input.clone()
        } else {
            input.clone()
        }
    }
}

#[derive(Clone)]
pub struct RandomCrop {
    size: (usize, usize),
    padding: Option<usize>,
}

impl RandomCrop {
    pub fn new(size: (usize, usize)) -> Self {
        Self { size, padding: None }
    }
    
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = Some(padding);
        self
    }
}

impl Transform for RandomCrop {
    fn apply(&self, input: &Tensor) -> Tensor {
        input.clone()
    }
}

#[derive(Clone)]
pub struct ToTensor;

impl ToTensor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ToTensor {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for ToTensor {
    fn apply(&self, input: &Tensor) -> Tensor {
        input.clone()
    }
}

pub fn train_test_split<D: Dataset>(
    dataset: D,
    test_ratio: f64,
    shuffle: bool,
) -> (Vec<usize>, Vec<usize>) {
    let len = dataset.len();
    let mut indices: Vec<usize> = (0..len).collect();

    if shuffle {
        indices.shuffle(&mut thread_rng());
    }

    let test_size = (len as f64 * test_ratio) as usize;
    let test_indices = indices[..test_size].to_vec();
    let train_indices = indices[test_size..].to_vec();

    (train_indices, test_indices)
}

pub fn xor_dataset() -> TensorDataset {
    let data = Tensor::from_slice(
        &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        &[4, 2],
    ).expect("Failed to create XOR data");

    let targets = Tensor::from_slice(
        &[0.0, 1.0, 1.0, 0.0],
        &[4],
    ).expect("Failed to create XOR targets");

    TensorDataset::new(data, targets).expect("Failed to create XOR dataset")
}

pub fn linear_regression_dataset(
    n_samples: usize,
    n_features: usize,
    noise: TensorDtype,
) -> TensorDataset {
    use rand::Rng;
    let mut rng = thread_rng();

    let mut data = Vec::with_capacity(n_samples * n_features);
    for _ in 0..n_samples * n_features {
        data.push(rng.gen::<TensorDtype>() * 10.0 - 5.0);
    }
    let data_tensor = Tensor::from_slice(&data, &[n_samples, n_features])
        .expect("Failed to create data tensor");

    let mut targets = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let row_sum: TensorDtype = (0..n_features)
            .map(|j| data[i * n_features + j])
            .sum();
        let noise_val = rng.gen::<TensorDtype>() * noise * 2.0 - noise;
        targets.push(row_sum + noise_val);
    }
    let target_tensor = Tensor::from_slice(&targets, &[n_samples])
        .expect("Failed to create target tensor");

    TensorDataset::new(data_tensor, target_tensor).expect("Failed to create dataset")
}