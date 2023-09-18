/*!
Implementation of Gabor filter.
 */

use crate::ops_2d;
use crate::tensor_ext::TensorExt;
use crate::tensor_init;
use tch::{Kind, Tensor};

/**
Generates a Gabor filter.
# Arguments
- size: usize - The size of the filter
- theta: f64 - The orientation of the filter in degrees
- sigma: f64 - The standard deviation of the Gaussian envelope
- lambda: f64 - The wavelength of the sinusoidal factor
- psi: f64 - The phase offset of the sinusoidal factor
- gamma: f64 - The spatial aspect ratio
- device: Device - The device to store the tensor on
# Returns
- [size, size] float tensor - The Gabor filter
 */
pub fn gabor_filter(
    size: usize,
    theta: f64,
    sigma: f64,
    lambda: f64,
    psi: f64,
    gamma: f64,
    device: tch::Device,
) -> Tensor {
    let xy = tensor_init::position_tensor_2d((size, size), 1, (Kind::Float, device));
    // Rotating the coordinates
    let xy_rot = ops_2d::rotate_2d(&xy, theta);
    // Scaling the coordinates
    let xy_rot_scaled = ops_2d::scale_2d(&xy_rot, &[gamma, 1.0]).squeeze();
    // Calculating the Gaussian envelope
    let gauss_env = xy_rot_scaled.square().sum_dim(-3) / (2.0 * sigma.powi(2));
    let gauss_env = gauss_env.neg().exp();
    // Calculating the sinusoidal factor
    let cos_factor = (2.0 * std::f64::consts::PI / lambda) * xy_rot_scaled.select(-3, 1) + psi;
    let cos_factor = cos_factor.cos();
    // Calculating the Gabor filter
    let gabor = gauss_env * cos_factor;
    gabor
}

/**
Applies Gabor filters to an input tensor.
> Note :
> The filters are using - 1.0 to 1.0 as range no matter the input size.
> Meaning not matter the input size, the filter will look the same.
# Arguments
- input: [N, 1, H, W] - The input tensor
- angle_count: usize - The number of angles to use
- filter_size: usize - The size of the filters
- frequencies: &[f64] - The frequencies to use
- sigma: f64 - The standard deviation of the Gaussian envelope
# Returns
- Tensor [N, angle_count * frequencies.len(), H, W] - The output tensor
    The C dimmension correspond to filters with different angles and frequencies.
    The filter are in the following order:
        - angle 0, frequency 0
        - angle 0, frequency 1
        - ...
        - angle 1, frequency 0
        - ...
 */
pub fn apply_gabor_filter(
    input: &Tensor,
    angle_count: usize,
    filter_size: usize,
    frequencies: &[f64],
    sigma: f64,
) -> Tensor {
    assert!(input.size().len() == 4);
    assert!(input.size()[1] == 1);
    let filters = (0..angle_count).flat_map(|i| {
        let theta = (i as f64) * std::f64::consts::PI / (angle_count as f64);
        frequencies.iter().map(move |&frequency| {
            let lambda = 1.0 / frequency;
            gabor_filter(filter_size, theta, sigma, lambda, 0.0, 1.0, input.device())
        })
        .collect::<Vec<_>>();
    let filters = Tensor::stack(&filters, 0).view([
        (angle_count * frequencies.len()) as i64,
        1,
        filter_size as i64,
        filter_size as i64,
    ]);
    input.conv2d_padding(&filters, None::<&Tensor>, &[1, 1], "same", &[1, 1], 1)
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use crate::ndarray::NDATensorExt;
    use ndarray_npy::write_npy;

    use super::*;
    use tch::index::*;

    #[test]
    fn test_gabor() {
        let gabor = gabor_filter(
            60,
            std::f64::consts::PI / 2.0,
            0.30,
            1.0,
            0.0,
            1.0,
            tch::Device::Cpu,
        );
        for i in 0..gabor.size()[0] {
            for j in 0..gabor.size()[1] {
                let gabor = f32::from(gabor.i((i, j)));
                let gabor = (gabor * 255.0 + 127.0) as u8;
                print!("\x1b[48;2;{};{};{}m  ", gabor, gabor, gabor);
            }
            println!("\x1b[0m");
        }
        for angle in 0..=7 {
            for frequencies in [0.5, 1.0, 2.0, 4.0, 6.0, 8.0] {
                let lambda = 1.0 / frequencies;
                let theta = (angle as f64) * PI / 8.0;
                let gabor = gabor_filter(60, theta, 0.45, lambda, 0.0, 1.0, tch::Device::Cpu);
                let gabor = gabor.to_ndarray();
                write_npy(
                    format!("test-results/gabor_{}_{}.npy", angle, frequencies),
                    &gabor,
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn test_apply_gabor_filter() {
        let input = Tensor::randn(&[2, 1, 60, 60], (Kind::Float, tch::Device::Cpu));
        let output = apply_gabor_filter(&input, 8, 31, &[0.06, 0.12, 0.24, 0.48], 0.50);
        assert!(output.size() == [2, 32, 60, 60]);
    }
}
