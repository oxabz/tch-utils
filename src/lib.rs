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
 * 
 * ## Conventions 
 * 
 * ### Shapes
 * - N : The number of samples
 * - C : The number of channels
 * - H : The height of the image
 * - W : The width of the image
 * 
 * - [N, C, H, W] : A tensor of shape [N, C, H, W] is a batch of N images of shape [C, H, W]
 *                 A tensor of multiple sample with one chanel will never be represented as [N, H, W] but as [N, 1, H, W]
 * - [C, H, W] : A tensor of shape [C, H, W] is an image of shape [C, H, W]
 * - [H, W] : A tensor of shape [H, W] is an image of shape [1, H, W]
 * 
 * *tensor of 2d vectors*
 * Tensors of 2d vectors ([N, 2, H, W] & [2, H, W]) are in the form [y, x]
 * 
 * ### Axis 
 * 
 * the y axis will always be top to bottom
 * the x axis will always be left to right
 * 
 * ```no_run
 * 0 ----> 1 (x)
 * |
 * |
 * v
 * 1 
 * ```
 * (y)
 * 
 * 
 */

pub mod ops_2d;
pub mod tensor_init;
pub mod noise;
pub mod shapes;
pub mod utils;