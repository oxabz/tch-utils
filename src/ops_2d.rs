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
 * ```rust
 * let a = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0]).view([1, 2, 2, 1]);
 * let b = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0]).view([1, 2, 2, 1]);
 * dot_product_2d(a, b);
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
  let typ = a.kind();
  a / (a.linalg_norm(2, vec![1].as_slice(), true, typ) + 1e-8)
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