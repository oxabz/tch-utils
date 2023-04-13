/*!
 * # 2D Operations
 * 
 * This module contains operations on tensor of shape [N, C, H, W] on each pixel.
 * 
 */

use tch::Tensor;

/**
 * Compute the vector dot product between two tensors for each pixel.
 *
 * # Arguments
 * a: Tensor - The first tensor [N, C, H, W]
 * b: Tensor - The second tensor [N, C, H, W]
 *
 * # Returns
 * Tensor - The dot product of the two tensors [N, 1 ,H, W]
 *
 * # Example
 * ```rust,no_run
 * # use tch::Tensor;
 * # use tch_utils::ops_2d::dot_product_2d;
 * let a = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0]).view([1, 2, 2, 1]);
 * let b = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0]).view([1, 2, 2, 1]);
 * let dot = dot_product_2d(&a, &b);
 * assert!(dot.equal(&Tensor::of_slice(&[10.0, 20.0]).view([1, 1, 2, 1])));
 * ```
 */
pub fn dot_product_2d(a: &Tensor, b: &Tensor) -> Tensor {
  let typ = a.kind();
  (a * b).sum_dim_intlist(vec![1].as_slice(), true, typ)
}

/**
 * Normalize vectors in 2d pictures to unit length.
 *
 * # Arguments
 * a: Tensor - The tensor to normalize [N, C, H, W]
 *
 * # Returns
 * Tensor - The normalized tensor [N, C, H, W]
 */
pub fn normalize_2d(a: &Tensor) -> Tensor {
  a / (norm_2d(a) + 1e-8)
}

/**
 * Compute the norm of vectors in 2d pictures. Equivalent to `linalg_norm` but faster.
 * 
 * # Arguments
 * - t: Tensor - The tensor to compute the norm [N, C, H, W]
 * 
 * # Returns
 * Tensor - The norm of the vectors [N, 1, H, W]
 */
pub fn norm_2d(t: &Tensor) -> Tensor {
  let typ = t.kind();
  (t.pow_tensor_scalar(2.0))
    .sum_dim_intlist(vec![1].as_slice(), true, typ)
    .sqrt()
}

/**
 * Scale the of vectors in 2d pictures.
 *
 * # Arguments
 * t: Tensor - The tensor to scale [N, C, H, W]
 * scale: &[f64] - The scale factors (lenght == C)
 *
 * # Returns
 * Tensor - The scaled tensor [N, C, H, W]
 */
pub fn scale_2d(t: &Tensor, scale: &[f64]) -> Tensor {
  let typ = t.kind();
  let device = t.device();
  let scale = Tensor::of_slice(scale)
    .to_kind(typ)
    .to_device(device)
    .view([1, scale.len() as i64, 1, 1]);
  t * scale
}

/**
 * Rotate the of vectors in 2d pictures.
 * 
 * # Arguments
 * t: Tensor - The tensor to rotate [N, 2, H, W]
 * angle: f64 - The angle of rotation in radians
 * 
 * # Returns
 * Tensor - The rotated tensor [N, 2, H, W]
 */
pub fn rotate_2d(t: &Tensor, angle: f64) -> Tensor {
  let typ = t.kind();
  let device = t.device();
  let cos = angle.cos();
  let sin = angle.sin();
  let rotation = Tensor::of_slice(&[cos, -sin, sin, cos])
    .to_kind(typ)
    .to_device(device)
    .view([2, 2]);
  
  t.transpose(1,3).matmul(&rotation).transpose(1,3)
}

/**
 * Translate the of vectors in 2d pictures.
 *
 * # Arguments
 * t: Tensor - The tensor to translate [N, C, H, W]
 * translation: &[f64] - The translation factors (lenght == C)
 *
 * # Returns
 * Tensor - The translated tensor [N, C, H, W]
 */
pub fn translate_2d(t: &Tensor, translation: &[f64]) -> Tensor {
  let typ = t.kind();
  let device = t.device();
  let translation = Tensor::of_slice(translation)
    .to_kind(typ)
    .to_device(device)
    .view([1, translation.len() as i64, 1, 1]);
  t + translation
}