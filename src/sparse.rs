use crate::tensor::{Tensor, TensorDtype};
use ndarray::{ArrayD, IxDyn, Dimension};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SparseTensorCOO {
    indices: Vec<Vec<usize>>,
    values: Vec<TensorDtype>,
    shape: Vec<usize>,
}

impl SparseTensorCOO {
    pub fn new(indices: Vec<Vec<usize>>, values: Vec<TensorDtype>, shape: Vec<usize>) -> Self {
        Self { indices, values, shape }
    }
    
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            indices: vec![],
            values: vec![],
            shape: shape.to_vec(),
        }
    }
    
    pub fn from_dense(tensor: &Tensor) -> Self {
        let shape = tensor.shape();
        let mut indices = Vec::new();
        let mut values = Vec::new();
        
        for (idx, &val) in tensor.data().indexed_iter() {
            if val != 0.0 {
                let index: Vec<usize> = idx.slice().iter().map(|&i| i).collect();
                indices.push(index);
                values.push(val);
            }
        }
        
        Self { indices, values, shape }
    }
    
    pub fn to_dense(&self) -> Tensor {
        let mut data = ArrayD::zeros(IxDyn(&self.shape));
        
        for (idx, &val) in self.indices.iter().zip(self.values.iter()) {
            let index_slice: Vec<ndarray::Ix> = idx.iter().map(|&i| i as ndarray::Ix).collect();
            data[&*index_slice] = val;
        }
        
        Tensor::from_array(data)
    }
    
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    
    pub fn density(&self) -> TensorDtype {
        let total: usize = self.shape.iter().product();
        self.nnz() as TensorDtype / total as TensorDtype
    }
    
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.shape != other.shape {
            return None;
        }
        
        let mut result_indices = self.indices.clone();
        let mut result_values = self.values.clone();
        
        for (idx, val) in other.indices.iter().zip(other.values.iter()) {
            if let Some(pos) = result_indices.iter().position(|i| i == idx) {
                result_values[pos] += val;
            } else {
                result_indices.push(idx.clone());
                result_values.push(*val);
            }
        }
        
        Some(Self::new(result_indices, result_values, self.shape.clone()))
    }
    
    pub fn mul_scalar(&self, scalar: TensorDtype) -> Self {
        let values: Vec<TensorDtype> = self.values.iter().map(|&v| v * scalar).collect();
        Self::new(self.indices.clone(), values, self.shape.clone())
    }
    
    pub fn transpose(&self) -> Self {
        if self.shape.len() != 2 {
            return self.clone();
        }
        
        let mut new_indices: Vec<Vec<usize>> = self.indices.iter()
            .map(|idx| vec![idx[1], idx[0]])
            .collect();
        
        let new_shape = vec![self.shape[1], self.shape[0]];
        
        Self::new(new_indices, self.values.clone(), new_shape)
    }
}

#[derive(Debug, Clone)]
pub struct SparseTensorCSR {
    data: Vec<TensorDtype>,
    indices: Vec<usize>,
    indptr: Vec<usize>,
    shape: (usize, usize),
}

impl SparseTensorCSR {
    pub fn new(data: Vec<TensorDtype>, indices: Vec<usize>, indptr: Vec<usize>, shape: (usize, usize)) -> Self {
        Self { data, indices, indptr, shape }
    }
    
    pub fn from_coo(coo: &SparseTensorCOO) -> Option<Self> {
        if coo.shape().len() != 2 {
            return None;
        }
        
        let rows = coo.shape()[0];
        let cols = coo.shape()[1];
        
        let mut row_indices: HashMap<usize, Vec<(usize, TensorDtype)>> = HashMap::new();
        
        for (idx, &val) in coo.indices.iter().zip(coo.values.iter()) {
            let row = idx[0];
            let col = idx[1];
            row_indices.entry(row).or_default().push((col, val));
        }
        
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];
        
        for row in 0..rows {
            if let Some(cols) = row_indices.get(&row) {
                let mut cols_sorted = cols.clone();
                cols_sorted.sort_by_key(|&(c, _)| c);
                for (col, val) in cols_sorted {
                    indices.push(col);
                    data.push(val);
                }
            }
            indptr.push(data.len());
        }
        
        Some(Self::new(data, indices, indptr, (rows, cols)))
    }
    
    pub fn to_dense(&self) -> Tensor {
        let mut data = ArrayD::zeros(IxDyn(&[self.shape.0, self.shape.1]));
        
        for row in 0..self.shape.0 {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            for j in start..end {
                let col = self.indices[j];
                data[[row, col]] = self.data[j];
            }
        }
        
        Tensor::from_array(data)
    }
    
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
    
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

pub fn sparse_matmul(a: &SparseTensorCOO, b: &SparseTensorCOO) -> Option<SparseTensorCOO> {
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return None;
    }
    
    if a.shape()[1] != b.shape()[0] {
        return None;
    }
    
    let a_dense = a.to_dense();
    let b_dense = b.to_dense();
    let result = a_dense.matmul(&b_dense).ok()?;
    
    Some(SparseTensorCOO::from_dense(&result))
}

pub fn sparse_dense_matmul(sparse: &SparseTensorCOO, dense: &Tensor) -> Option<Tensor> {
    let sparse_dense = sparse.to_dense();
    sparse_dense.matmul(dense).ok()
}
