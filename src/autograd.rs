use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::error::{LeorchError, Result};
use crate::tensor::{Tensor, TensorDtype};

pub type NodeId = usize;

static mut NEXT_NODE_ID: NodeId = 0;

fn get_next_node_id() -> NodeId {
    unsafe {
        let id = NEXT_NODE_ID;
        NEXT_NODE_ID += 1;
        id
    }
}

pub trait Node: fmt::Debug {
    fn backward(&self, grad_output: &ArrayD<TensorDtype>) -> Vec<ArrayD<TensorDtype>>;
    fn num_inputs(&self) -> usize;
    fn op_name(&self) -> &str;
}

#[derive(Clone)]
pub struct Variable {
    pub data: ArrayD<TensorDtype>,
    pub grad: Option<Rc<RefCell<ArrayD<TensorDtype>>>>,
    pub grad_fn: Option<Rc<dyn Node>>,
    pub requires_grad: bool,
    pub id: NodeId,
}

impl Variable {
    pub fn from_tensor(tensor: &Tensor) -> Self {
        Self {
            data: tensor.data.clone(),
            grad: if tensor.requires_grad {
                Some(Rc::new(RefCell::new(ArrayD::zeros(tensor.data.shape()))))
            } else {
                None
            },
            grad_fn: None,
            requires_grad: tensor.requires_grad,
            id: get_next_node_id(),
        }
    }
    
    pub fn new(data: ArrayD<TensorDtype>, requires_grad: bool) -> Self {
        Self {
            data: data.clone(),
            grad: if requires_grad {
                Some(Rc::new(RefCell::new(ArrayD::zeros(data.shape()))))
            } else {
                None
            },
            grad_fn: None,
            requires_grad,
            id: get_next_node_id(),
        }
    }
    
    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        let data = ArrayD::zeros(IxDyn(shape));
        Self::new(data, requires_grad)
    }
    
    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        let data = ArrayD::ones(IxDyn(shape));
        Self::new(data, requires_grad)
    }
    
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }
    
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }
    
    pub fn zero_grad(&mut self) {
        if let Some(ref grad) = self.grad {
            *grad.borrow_mut() = ArrayD::zeros(self.data.shape());
        }
    }
    
    pub fn backward(&mut self) -> Result<()> {
        if !self.requires_grad {
            return Err(LeorchError::GradientError(
                "Cannot call backward on a variable that doesn't require gradients".to_string()
            ));
        }
        if let Some(ref grad) = self.grad {
            *grad.borrow_mut() = ArrayD::ones(self.data.shape());
        }
        if let Some(ref grad_fn) = self.grad_fn {
            self.backward_recursive(grad_fn, &ArrayD::ones(self.data.shape()))?;
        }
        Ok(())
    }
    
    fn backward_recursive(
        &self,
        node: &Rc<dyn Node>,
        grad_output: &ArrayD<TensorDtype>,
    ) -> Result<()> {
        let _grads = node.backward(grad_output);
        Ok(())
    }
    
    pub fn to_tensor(&self) -> Tensor {
        Tensor::from_array(self.data.clone())
    }
    
    pub fn add(&self, other: &Variable) -> Variable {
        let result_data = &self.data + &other.data;
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);
        if requires_grad {
            result.grad_fn = Some(Rc::new(AddBackward {
                input_shapes: vec![self.shape(), other.shape()],
            }));
        }
        result
    }
    
    pub fn mul(&self, other: &Variable) -> Variable {
        let result_data = &self.data * &other.data;
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);
        if requires_grad {
            result.grad_fn = Some(Rc::new(MulBackward {
                input_shapes: vec![self.shape(), other.shape()],
                saved_inputs: vec![self.data.clone(), other.data.clone()],
            }));
        }
        result
    }
    
    pub fn matmul(&self, other: &Variable) -> Result<Variable> {
        let self_2d = self.data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let other_2d = other.data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let result_data = self_2d.dot(&other_2d).into_dyn();
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);
        if requires_grad {
            result.grad_fn = Some(Rc::new(MatmulBackward {
                a_shape: self.shape(),
                b_shape: other.shape(),
                saved_a: self.data.clone(),
                saved_b: other.data.clone(),
            }));
        }
        Ok(result)
    }
    
    pub fn pow(&self, exponent: TensorDtype) -> Variable {
        let result_data = self.data.mapv(|x| x.powf(exponent));
        let mut result = Variable::new(result_data, self.requires_grad);
        if self.requires_grad {
            result.grad_fn = Some(Rc::new(PowBackward {
                exponent,
                saved_input: self.data.clone(),
            }));
        }
        result
    }
    
    pub fn sum(&self) -> Variable {
        let result_data = ArrayD::from_elem(ndarray::IxDyn(&[]), self.data.sum());
        let mut result = Variable::new(result_data, self.requires_grad);
        if self.requires_grad {
            result.grad_fn = Some(Rc::new(SumBackward {
                input_shape: self.shape(),
            }));
        }
        result
    }
    
    pub fn mean(&self) -> Variable {
        let n = self.data.len() as TensorDtype;
        let result_data = ArrayD::from_elem(ndarray::IxDyn(&[]), self.data.sum() / n);
        let mut result = Variable::new(result_data, self.requires_grad);
        if self.requires_grad {
            result.grad_fn = Some(Rc::new(MeanBackward {
                input_shape: self.shape(),
                n,
            }));
        }
        result
    }
    
    pub fn relu(&self) -> Variable {
        let result_data = self.data.mapv(|x| x.max(0.0));
        let mut result = Variable::new(result_data, self.requires_grad);
        if self.requires_grad {
            result.grad_fn = Some(Rc::new(ReLUBackward {
                saved_input: self.data.clone(),
            }));
        }
        result
    }
    
    pub fn sigmoid(&self) -> Variable {
        let result_data = self.data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let mut result = Variable::new(result_data.clone(), self.requires_grad);
        if self.requires_grad {
            result.grad_fn = Some(Rc::new(SigmoidBackward {
                saved_output: result_data,
            }));
        }
        result
    }
    
    pub fn tanh(&self) -> Variable {
        let result_data = self.data.mapv(|x| x.tanh());
        let mut result = Variable::new(result_data.clone(), self.requires_grad);
        if self.requires_grad {
            result.grad_fn = Some(Rc::new(TanhBackward {
                saved_output: result_data,
            }));
        }
        result
    }
}

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Variable")
            .field("shape", &self.shape())
            .field("requires_grad", &self.requires_grad)
            .field("data", &self.data)
            .field("grad_fn", &self.grad_fn.as_ref().map(|_| "Some(Node)"))
            .finish()
    }
}

#[derive(Debug)]
struct AddBackward {
    input_shapes: Vec<Vec<usize>>,
}

impl Node for AddBackward {
    fn backward(&self, grad_output: &ArrayD<TensorDtype>) -> Vec<ArrayD<TensorDtype>> {
        vec![grad_output.clone(), grad_output.clone()]
    }
    
    fn num_inputs(&self) -> usize {
        2
    }
    
    fn op_name(&self) -> &str {
        "Add"
    }
}

#[derive(Debug)]
struct MulBackward {
    input_shapes: Vec<Vec<usize>>,
    saved_inputs: Vec<ArrayD<TensorDtype>>,
}

impl Node for MulBackward {
    fn backward(&self, grad_output: &ArrayD<TensorDtype>) -> Vec<ArrayD<TensorDtype>> {
        let x = &self.saved_inputs[0];
        let y = &self.saved_inputs[1];
        vec![
            grad_output * y,
            grad_output * x,
        ]
    }
    
    fn num_inputs(&self) -> usize {
        2
    }
    
    fn op_name(&self) -> &str {
        "Mul"
    }
}

#[derive(Debug)]
struct MatmulBackward {
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    saved_a: ArrayD<TensorDtype>,
    saved_b: ArrayD<TensorDtype>,
}

impl Node for MatmulBackward {
    fn backward(&self, grad_output: &ArrayD<TensorDtype>) -> Vec<ArrayD<TensorDtype>> {
        let grad_output_2d = grad_output.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let saved_b_2d = self.saved_b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let saved_a_2d = self.saved_a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        
        let grad_a = grad_output_2d.dot(&saved_b_2d.t()).into_dyn();
        let grad_b = saved_a_2d.t().dot(&grad_output_2d).into_dyn();
        
        vec![grad_a, grad_b]
    }
    
    fn num_inputs(&self) -> usize {
        2
    }
    
    fn op_name(&self) -> &str {
        "Matmul"
    }
}

#[derive(Debug)]
struct PowBackward {
    exponent: TensorDtype,
    saved_input: ArrayD<TensorDtype>,
}

impl Node for PowBackward {
    fn backward(&self, grad_output: &ArrayD<TensorDtype>) -> Vec<ArrayD<TensorDtype>> {
        let grad = grad_output * self.saved_input.mapv(|x| {
            self.exponent * x.powf(self.exponent - 1.0)
        });
        vec![grad]
    }
    
    fn num_inputs(&self) -> usize {
        1
    }
    
    fn op_name(&self) -> &str {
        "Pow"
    }
}

#[derive(Debug)]
struct SumBackward {
    input_shape: Vec<usize>,
}

impl Node for SumBackward {
    fn backward(&self, grad_output: &ArrayD<TensorDtype>) -> Vec<ArrayD<TensorDtype>> {
        let grad = ArrayD::ones(IxDyn(&self.input_shape)) * grad_output.first().copied().unwrap_or(1.0);
        vec![grad]
    }
    
    fn num_inputs(&self) -> usize {
        1
    }
    
    fn op_name(&self) -> &str {
        "Sum"
    }
}

#[derive(Debug)]
struct MeanBackward {
    input_shape: Vec<usize>,
    n: TensorDtype,
}

impl Node for MeanBackward {
    fn backward(&self, grad_output: &ArrayD<TensorDtype>) -> Vec<ArrayD<TensorDtype>> {
        let grad = ArrayD::ones(IxDyn(&self.input_shape)) * grad_output.first().copied().unwrap_or(1.0) / self.n;
        vec![grad]
    }
    
    fn num_inputs(&self) -> usize {
        1
    }
    
    fn op_name(&self) -> &str {
        "Mean"
    }
}

#[derive(Debug)]
struct ReLUBackward {
    saved_input: ArrayD<TensorDtype>,
}

impl Node for ReLUBackward {
    fn backward(&self, grad_output: &ArrayD<TensorDtype>) -> Vec<ArrayD<TensorDtype>> {
        let grad = grad_output * self.saved_input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        vec![grad]
    }
    
    fn num_inputs(&self) -> usize {
        1
    }
    
    fn op_name(&self) -> &str {
        "ReLU"
    }
}

#[derive(Debug)]
struct SigmoidBackward {
    saved_output: ArrayD<TensorDtype>,
}

impl Node for SigmoidBackward {
    fn backward(&self, grad_output: &ArrayD<TensorDtype>) -> Vec<ArrayD<TensorDtype>> {
        let s = &self.saved_output;
        let grad = grad_output * s * s.mapv(|x| 1.0 - x);
        vec![grad]
    }
    
    fn num_inputs(&self) -> usize {
        1
    }
    
    fn op_name(&self) -> &str {
        "Sigmoid"
    }
}

#[derive(Debug)]
struct TanhBackward {
    saved_output: ArrayD<TensorDtype>,
}

impl Node for TanhBackward {
    fn backward(&self, grad_output: &ArrayD<TensorDtype>) -> Vec<ArrayD<TensorDtype>> {
        let t = &self.saved_output;
        let grad = grad_output * t.mapv(|x| 1.0 - x * x);
        vec![grad]
    }
    
    fn num_inputs(&self) -> usize {
        1
    }
    
    fn op_name(&self) -> &str {
        "Tanh"
    }
}
