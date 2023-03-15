/*!
 * # Noise
 * 
 * This module contains noise functions 
 */

use rand::{SeedableRng, Rng, distributions};
use tch::Tensor;

use crate::{tensor_init::*, ops_2d::*};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Corner {
  TopLeft,
  TopRight,
  BottomLeft,
  BottomRight,
}

fn compute_corner_contribution(
  pos: &Tensor,
  grid: &Tensor,
  offset: &Tensor,
  corner: Corner,
  (res_x, res_y): (usize, usize),
) -> Tensor {
  let typ = pos.kind();
  let device = pos.device();

  let corner_offset = match corner {
    Corner::TopLeft => Tensor::of_slice(&[0.0, 0.0]).view([1, 2, 1, 1]),
    Corner::TopRight => Tensor::of_slice(&[0.0, 1.0]).view([1, 2, 1, 1]),
    Corner::BottomLeft => Tensor::of_slice(&[1.0, 0.0]).view([1, 2, 1, 1]),
    Corner::BottomRight => Tensor::of_slice(&[1.0, 1.0]).view([1, 2, 1, 1]),
  }
  .to_device(device)
  .to_kind(typ);

  let magic_numbers = match corner {
    Corner::TopLeft => Tensor::of_slice(&[1.0, 1.0]).view([1, 2, 1, 1]),
    Corner::TopRight => Tensor::of_slice(&[1.0, -1.0]).view([1, 2, 1, 1]),
    Corner::BottomLeft => Tensor::of_slice(&[-1.0, 1.0]).view([1, 2, 1, 1]),
    Corner::BottomRight => Tensor::of_slice(&[-1.0, -1.0]).view([1, 2, 1, 1]),
  }
  .to_device(device)
  .to_kind(typ);

  let sample_hr = {
    // [N, 2, H, W] -> [N, H, W ,2]
    let pos = pos.swapaxes(1, 3).swapaxes(1, 2);
    let pos = pos
      + corner_offset.view([1, 1, 1, 2])
        * Tensor::of_slice(&[2.0 / res_y as f64, 2.0 / res_x as f64])
          .to_device(device)
          .to_kind(typ)
          .view([1, 1, 1, 2]);
    grid.grid_sampler(&pos, 1, 2, false)
  };

  let offset = offset - corner_offset;

  let weight = (magic_numbers - offset.shallow_clone()).prod_dim_int(1, true, tch::Kind::Float)
    * if corner == Corner::TopLeft || corner == Corner::BottomRight {
      1.0
    } else {
      -1.0
    };

  let offset = normalize_2d(&offset);

  dot_product_2d(&offset, &sample_hr) * weight
}


/**
 * Generate 2d perlin noise.
 *
 * # Arguments
 * shape: (usize, usize) - The shape of the resulting image
 * n: usize - The number samples generated
 * res: (usize, usize) - The resolution of the noise
 * seed: u64 - The seed for the random number generator
 * options: (tch::Kind, tch::Device) - The kind and device of the resulting tensor
 *
 * # Returns
 * Tensor - The perlin noise [N, 1, H, W]
 */
pub fn perlin_noise_2d(
  shape: (usize, usize),
  n: usize,
  res: (usize, usize),
  seed: u64,
  options: (tch::Kind, tch::Device),
) -> Tensor {
  tch::no_grad_guard();
  let (w, h) = shape;
  let (res_x, res_y) = res;
  let (kind, device) = options.clone();

  // Generate random directions [N, 2, H, W]
  let rng = rand::rngs::StdRng::seed_from_u64(seed);
  let grid = rng
    .sample_iter(distributions::Uniform::new(-1.0, 1.0))
    .take(res_x * res_y * 2 * n)
    .collect::<Vec<f64>>();

  let grid = Tensor::of_slice(&grid)
    .view([n as i64, 2, res_x as i64, res_y as i64])
    .to_device(device)
    .to_kind(kind);
  let grid = normalize_2d(&grid);

  // Generate a tensor of positions [N, 2, H, W]
  let pos = position_tensor_2d((w, h), n, (tch::Kind::Float, device));
  let offset_x = 1.0 / w as f64;
  let offset_y = 1.0 / h as f64;
  let pos = translate_2d(&pos, &[offset_y, offset_x]);

  // Generate a tensor of offsets [N, 2, H, W]
  let offset = {
    let pos = pos.shallow_clone();
    let pos = translate_2d(&pos, &[1.0, 1.0]);
    let pos = scale_2d(&pos, &[0.5 * res_y as f64, 0.5 * res_x as f64]);
    let ipos = pos.floor();
    pos - ipos
  };

  let mut res = compute_corner_contribution(&pos, &grid, &offset, Corner::TopLeft, (res_x, res_y));
  res += compute_corner_contribution(&pos, &grid, &offset, Corner::TopRight, (res_x, res_y));
  res += compute_corner_contribution(&pos, &grid, &offset, Corner::BottomLeft, (res_x, res_y));
  res += compute_corner_contribution(&pos, &grid, &offset, Corner::BottomRight, (res_x, res_y));

  res
}
