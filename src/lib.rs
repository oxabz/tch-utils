/*!
 * # Tch-utils - A collection of utilities for the tch-rs crate
 * 
 * > *Note :* This crate is mostly intended for my own to collect some utilities I use in my projects.
 * > It is not intended to be used by anyone else. However, if you find it useful, feel free to use it. 
 * 
 * ## Features
 * - 2D operations : operations on tensor of shape [N, C, H, W] on each pixel
 * - Tensor initialization : new way to initialize tensors
 * - Noises : Generate noise tensors
 */

pub mod ops_2d;
pub mod tensor_init;
pub mod noise;
pub mod shapes;