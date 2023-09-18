/*!
# Tensor initialization functions

This module contains functions to initialize tensors that are not in the tch create.
 */

use tch::Tensor;

/**
Generate a tensor that contains a vec2 from (-1, -1) to (1, 1) depending on the position.

# Arguments
size: (usize, usize) - The size of the tensor
n: usize - The number of samples
options: (tch::Kind, tch::Device) - The kind and device of the tensor

# Returns
Tensor - The tensor containing the positions [N, 2, H, W]
The 2nd dimension contains the x and y position with y being the first element
# Example
```rust,no_run
# use tch::Tensor;
# use tch_utils::tensor_init::position_tensor_2d;
let pos = position_tensor_2d((3, 3), 1, (tch::Kind::Float, tch::Device::Cpu));
#[rustfmt::skip]
let expected = Tensor::of_slice(&[
 -1.0, -1.0, -1.0,
 0.0, 0.0, 0.0,
 1.0, 1.0, 1.0,
 -1.0, 0.0, 1.0,
 -1.0, 0.0, 1.0,
 -1.0, 0.0, 1.0,
]).view([1, 2, 3, 3]);
eprintln!("pos: {:?}", Vec::<f64>::from(&pos));
assert!(f64::from((pos - expected).abs().sum(tch::Kind::Float)) < 1e-6);
```
 */
pub fn position_tensor_2d(
    size: (usize, usize),
    n: usize,
    options: (tch::Kind, tch::Device),
) -> Tensor {
    let (w, h) = size;
    let (kind, device) = options.clone();

    let pos_x = Tensor::arange(w as i64, options);
    let pos_x = pos_x.repeat(&[h as i64, 1]);

    let pos_y = Tensor::arange(h as i64, options);
    let pos_y = pos_y.repeat(&[w as i64, 1]).transpose(0, 1);

    let pos = Tensor::stack(&[pos_y, pos_x], 0)
        / Tensor::of_slice(&[(h - 1) as f64 * 0.5, (w - 1) as f64 * 0.5])
            .to_device(device)
            .to_kind(kind)
            .view([2, 1, 1]);
    let pos = pos
        - Tensor::of_slice(&[1.0, 1.0])
            .to_device(device)
            .to_kind(kind)
            .view([2, 1, 1]);
    pos.unsqueeze(0).repeat(&[n as i64, 1, 1, 1])
}
