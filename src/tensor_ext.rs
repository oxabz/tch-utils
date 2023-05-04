/*!
Contains diverse extensions to the Tensor struct.
 */

use tch::Tensor;

pub trait TensorExt{
    fn sum_dim(&self, dim: i64) -> Tensor;
    fn sum_kdim(&self, dim: i64) -> Tensor;
    fn sum_dims<const D: usize>(&self, dims: [i64;D]) -> Tensor;
    fn sum_kdims<const D: usize>(&self, dims: [i64;D]) -> Tensor;
}

impl TensorExt for Tensor {
    fn sum_dim(&self, dim: i64) -> Tensor {
        let typ = self.kind(); 
        self.sum_dim_intlist(Some(&[dim][..]), false, typ)
    }

    fn sum_kdim(&self, dim: i64) -> Tensor {
        let typ = self.kind(); 
        self.sum_dim_intlist(Some(&[dim][..]), true, typ)
    }

    fn sum_dims<const D: usize>(&self, dims: [i64; D]) -> Tensor {
        let typ = self.kind();
        self.sum_dim_intlist(Some(&dims[..]), false, typ)
    }

    fn sum_kdims<const D: usize>(&self, dims: [i64; D]) -> Tensor {
        let typ = self.kind();
        self.sum_dim_intlist(Some(&dims[..]), true, typ)
    }
}