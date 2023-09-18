use tch::{index::*, Kind, Tensor};

const GLCM_BINCOUNT_SIZE: i64 = 0x00FF_FFFF;

/**
Computes the gray-level co-occurrence matrix (GLCM) of an image.

The GLCM is a 2D histogram of the co-occurrence of pixel values at a given offset over an image.

# Arguments
- image: Tensor of shape [N, 1, H, W] where N is the batch size, H and W are the height and width of the image.
    The values are expected to be in the range [0, 1].
- offset: The offset of the co-occurrence. The first element is the vertical offset and the second element is the horizontal offset.
- num_shades: The number of shades of gray to use for the GLCM.
    The values are expected to be in the range [2, 254]
    Internally the image will be cast from float to u8 and the value 255 is used to represent masked pixels.
- mask: A mask of shape [N, 1, H, W] where N is the batch size, H and W are the height and width of the image.
    The values are expected to be in the range [0, 1]. If provided, the GLCM will only be computed for the pixels where the mask is 1.
- symmetric: Whether to use a symmetric GLCM. If true, the GLCM will be computed for both the offset and its opposite.
    The final GLCM will be the sum of the two GLCMs.
    If false, only the GLCM for the offset will be computed.

# Returns
- Tensor of shape [N, num_shades, num_shades] where N is the batch size.
    The values are in the range [0, 1] and represent the normalized GLCM.
    dim 1 is the reference shade and dim 2 is the neighbor shade.

 */
pub fn glcm(
    image: &Tensor,
    offset: (i64, i64),
    num_shades: u8,
    mask: Option<&Tensor>,
    symmetric: bool,
) -> Tensor {
    if image.device().is_cuda() {
        glcm_gpu(image, offset, num_shades, mask, symmetric)
    } else {
        glcm_cpu(image, offset, num_shades, mask, symmetric)
    }
}

pub fn glcm_gpu(
    image: &Tensor,
    offset: (i64, i64),
    num_shades: u8,
    mask: Option<&Tensor>,
    symmetric: bool,
) -> Tensor {
    let (offset_y, offset_x) = offset;
    let (offset_y, offset_x) = (offset_y, offset_x);
    let (batch_size, _, height, width) = image.size4().unwrap();

    let mut image = image * num_shades as f64;
    if let Some(mask) = mask {
        image *= mask;
        let mask = (mask - 1.0) * -(num_shades as f64);
        image += mask;
    }
    let image = image.floor().clamp(0.0, 255.0).to_kind(Kind::Uint8);

    // preping the slices for the image
    let rslice = (
        ..,
        ..,
        (-offset_y).max(0)..(height - offset_y).min(height),
        (-offset_x).max(0)..(width - offset_x).min(width),
    );
    let nslice = (
        ..,
        ..,
        offset_y.max(0)..(height + offset_y).min(height),
        offset_x.max(0)..(width + offset_x).min(width),
    );

    // We take a shifted version of the image and compute the GLCM for each pixel.
    let ref_img = image.i(rslice.clone()).to_kind(Kind::Int64);
    let neigh_img = image.i(nslice.clone()).to_kind(Kind::Int64);

    // We then encode the values into a single number.
    // Saddly libtorch doesn't support uint32 tensors so we have to use int32.
    // We map the neighbor's value to the first 8 bits and the reference's value to the next 8 bits.
    // We then use an other 8 bits to encode the batch index we coulduse 15 bit for the batch but it would result in a gigabyte sized tensor.
    let num_shades = (num_shades + 1) as i64; // Number of shades + 1 for the masked pixels
    let group_size = ((GLCM_BINCOUNT_SIZE - 1) / num_shades.pow(2)).min(batch_size);
    let group_count = batch_size / group_size + (batch_size % group_size != 0) as i64;
    let group_size = batch_size / group_count + (batch_size % group_count != 0) as i64;
    let batch_idx =
        Tensor::arange(batch_size as i64, (Kind::Int64, image.device())).remainder(group_size);
    let batch_idx = batch_idx.view([-1, 1, 1, 1]);
    let pairs = batch_idx * num_shades.pow(2) + ref_img * num_shades + neigh_img;

    let glcms = {
        // We split the tensor into groups of size group_size.
        let pairs = pairs.tensor_split(group_count, 0);
        pairs
            .iter()
            .map(|t| (t.size()[0], t))
            .map(|(s, t)| (s, t.view([-1])))
            .map(|(s, t)| (s, t.bincount::<&Tensor>(None, num_shades.pow(2) * s)))
            .map(|(s,t)| t.view([s, num_shades, num_shades]))
            .collect::<Vec<_>>()
    };

    let glcm = Tensor::cat(&glcms[..], 0);
    let mut glcm = glcm.i((.., ..num_shades - 1, ..num_shades - 1));

    if symmetric {
        glcm += glcm.copy().transpose(-1, -2);
    }

    let len = glcm.sum_dim_intlist(Some(&[1, 2][..]), false, Kind::Float);
    let len = len.view([-1, 1, 1]);
    &glcm / len
}

pub fn glcm_cpu(
    image: &Tensor,
    offset: (i64, i64),
    num_shades: u8,
    mask: Option<&Tensor>,
    symmetric: bool,
) -> Tensor {
    let (offset_y, offset_x) = offset;

    let (batch_size, _, height, width) = image.size4().unwrap();
    let (batch_size, height, width) = (batch_size as usize, height as usize, width as usize);
    let device = image.device();

    let mut image = image * num_shades as f64;
    // If we use a mask we add 255 to the masked pixels so that they are not counted in the GLCM.
    if let Some(mask) = mask {
        let mask = (mask - 1.0) * -255.0;
        image += mask;
    }
    let image = image.clamp(0.0, 255.0).to_kind(Kind::Uint8);

    let batch_span = height * width;

    // Moving from Tensor to Vec<u8>
    let image = Vec::<u8>::from(&image);
    let it = ((-offset_y).max(0)..(height as i64 - offset_y).min(height as i64)).flat_map(|y| {
        ((-offset_x).max(0)..(width as i64 - offset_x).min(width as i64))
            .map(move |x| (y as usize, x as usize))
    });

    let glcm = (0..batch_size).map(|batch| {
        let mut glcm = vec![0; num_shades as usize * num_shades as usize];
        for (y, x) in it.clone() {
            let reference_shade = image[batch * batch_span + y * width + x] as usize;
            let neighbor_shade = image[batch * batch_span
                + (y as i64 + offset_y) as usize * width
                + (x as i64 + offset_x) as usize] as usize;
            if reference_shade >= num_shades as usize || neighbor_shade >= num_shades as usize {
                continue;
            }
            glcm[reference_shade * num_shades as usize + neighbor_shade] += 1;
        }
        Tensor::of_slice(&glcm).view([num_shades as i64, num_shades as i64])
    });
    let mut glcm = Tensor::stack(&glcm.collect::<Vec<_>>()[..], 0).to_device(device);
    if symmetric {
        glcm += glcm.copy().transpose(-1, -2);
    }
    let len = glcm.sum_dim_intlist(Some(&[1, 2][..]), false, Kind::Float);
    let len = len.view([-1, 1, 1]);
    glcm / len
}

pub mod features {
    use tch::{index::*, Device, Kind, Tensor};

    use crate::tensor_ext::TensorExt;

    /**
    The features computed from a GLCM.
     */
    pub struct GlcmFeatures {
        pub correlation: Tensor,
        pub contrast: Tensor,
        pub dissimilarity: Tensor,
        pub entropy: Tensor,
        pub angular_second_moment: Tensor,
        pub sum_average: Tensor,
        pub sum_variance: Tensor,
        pub sum_entropy: Tensor,
        pub sum_of_squares: Tensor,
        pub inverse_difference_moment: Tensor,
        pub difference_average: Tensor,
        pub difference_variance: Tensor,
        pub information_measure_of_correlation_1: Tensor,
        pub information_measure_of_correlation_2: Tensor,
    }

    /**
    Compute the elements used to compute the GLCM features.
     */
    fn preludes(
        glcm: &Tensor,
        batch_size: i64,
        num_shades: i64,
        tensor_options: (Kind, Device),
    ) -> (
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
    ) {
        let px = glcm.sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N, num_shades]
        let py = glcm.sum_dim_intlist(Some(&[-2][..]), false, Kind::Float); // [N, num_shades]

        let levels = Tensor::arange(num_shades, (Kind::Float, glcm.device())).unsqueeze(0); // [1, num_shades]
        let u_x = (&px * &levels).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]
        let u_y = (&py * &levels).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]

        let (pxpy, pxdy) = {
            // [N, LEVEL * 2 - 2], [N, LEVEL]
            let pxpy = Tensor::zeros(&[batch_size, num_shades as i64 * 2 - 2], tensor_options); // [N, LEVEL * 2 - 2]
            let pxdy = Tensor::zeros(&[batch_size, num_shades as i64], tensor_options); // [N, LEVEL]
            for i in 0..num_shades {
                for j in 0..num_shades {
                    let (i, j) = (i as i64, j as i64);
                    let idx1 = (i + j) - 2;
                    let idx2 = (i - j).abs();
                    let mut t1 = pxpy.i((.., idx1));
                    let mut t2 = pxdy.i((.., idx2));
                    t1 += glcm.i((.., i, j));
                    t2 += glcm.i((.., i, j));
                }
            }
            (pxpy, pxdy)
        };

        let entropy_x =
            -(&px * (&px + 1e-6).log2()).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]
        let entropy_y =
            -(&py * (&py + 1e-6).log2()).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]
        let entropy_xy =
            -(glcm * (glcm + 1e-6).log2()).sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float); // [N]

        let (hxy1, hxy2) = {
            // [N], [N]
            let pxpy = px.unsqueeze(-1).matmul(&py.unsqueeze(-2)); // [N, LEVEL, LEVEL]
            let hxy1 = -(glcm * (&pxpy + 1e-6).log2()).sum_dim_intlist(
                Some(&[-1, -2][..]),
                false,
                Kind::Float,
            );
            let hxy2 = -(&pxpy * (&pxpy + 1e-6).log2()).sum_dim_intlist(
                Some(&[-1, -2][..]),
                false,
                Kind::Float,
            );
            (hxy1, hxy2)
        };

        (
            u_x, u_y, pxpy, pxdy, entropy_x, entropy_y, entropy_xy, hxy1, hxy2,
        )
    }

    /**
    Computes the GLCM features of a GLCM.

    # Arguments
    - glcm: Tensor of shape [N, num_shades, num_shades] where N is the batch size.
        The values are in the range [0, 1] and represent the normalized GLCM.
        dim 1 is the reference shade and dim 2 is the neighbor shade.

    # Returns
    - GlcmFeatures: A struct containing the GLCM features.

     */
    pub fn glcm_features(glcm: &Tensor) -> GlcmFeatures {
        let (batch_size, num_shades, _) = glcm.size3().unwrap();
        let tensor_option = (glcm.kind(), glcm.device());

        let (u_x, u_y, pxpy, pxdy, entropy_x, entropy_y, entropy_xy, hxy1, hxy2) =
            preludes(&glcm, batch_size, num_shades, tensor_option);

        let correlation = {
            //[N]
            let intensity = Tensor::arange(num_shades as i64, tensor_option);
            let intensity = intensity
                .unsqueeze(1)
                .matmul(&intensity.unsqueeze(0))
                .unsqueeze(0); // [1, LEVEL, LEVEL]
            (glcm * intensity - (&u_x * &u_y).view([-1, 1, 1])).sum_dims([-1, -2])
        };

        let (contrast, dissimilarity) = {
            let i = Tensor::arange(num_shades, tensor_option).view([1, 1, -1]);
            let j = Tensor::arange(num_shades, tensor_option).view([1, -1, 1]);

            // Here we compute the features together to avoid allocating imj twice.
            // We compute the absolute value for the dissimilarity
            // Then we square the difference for the contrast we can do this because x² = |x|²
            let mut imj = &i - &j; // [1, LEVEL, LEVEL]
            let mut imj = imj.abs_();
            let dissimilarity = (glcm * &imj).sum_dims([-1, -2]);
            let mut imj = imj.square_(); // [1, LEVEL, LEVEL]
            imj *= i;
            let contrast = (glcm * imj).sum_dims([-1, -2]); // [N]
            (contrast, dissimilarity)
        };

        let entropy = -(glcm * (glcm + 1e-6).log2()).sum_dims([-1, -2]); // [N]

        let angular_second_moment = glcm.square().sum_dims([-1, -2]); // [N]

        let (sum_average, sum_variance) = {
            let k = Tensor::arange(2 * num_shades - 2, tensor_option).unsqueeze(0);
            let sum_average = (&k * &pxpy).sum_kdim(-1); // [N, 1]
            let sum_variance = ((k - &sum_average).square() * &pxpy).sum_dim(-1); // [N]

            (sum_average.squeeze(), sum_variance)
        };

        let sum_entropy = -(&pxpy * (&pxpy + 1e-6).log2()).sum_dim(-1); // [N]

        let sum_of_squares = {
            let i = Tensor::arange(num_shades, tensor_option).view([1, 1, -1]);
            let mut i = i - &u_x.view([-1, 1, 1]);
            let i = i.square_();

            (i * glcm).sum_dims([-1, -2])
        };

        let inverse_difference_moment = {
            let i = Tensor::arange(num_shades, tensor_option).view([1, 1, -1]);
            let j = Tensor::arange(num_shades, tensor_option).view([1, -1, 1]);
            let mut imj = i - j;
            let mut imj = imj.square_();
            imj += 1.0;
            (glcm / imj).sum_dims([-1, -2])
        };

        let (difference_average, difference_variance) = {
            let k = Tensor::arange(num_shades, tensor_option).unsqueeze(0);
            let difference_average = (&k * &pxdy).sum_kdim(-1); // [N, 1]
            let difference_variance = ((k - &difference_average).square() * &pxdy).sum_dim(-1); // [N]

            (difference_average.squeeze(), difference_variance)
        };

        let information_measure_of_correlation_1 =
            (&entropy_xy - hxy1) / entropy_x.max_other(&entropy_y);
        let information_measure_correlation2 = (-((hxy2 - &entropy_xy) * -2.0).exp() + 1.0).sqrt();

        GlcmFeatures {
            correlation,
            contrast,
            dissimilarity,
            entropy,
            angular_second_moment,
            sum_average,
            sum_variance,
            sum_entropy,
            sum_of_squares,
            inverse_difference_moment,
            difference_average,
            difference_variance,
            information_measure_of_correlation_1,
            information_measure_of_correlation_2: information_measure_correlation2,
        }
    }
}

#[cfg(test)]
mod test {
    use tch::{index::*, Device, Kind, Tensor};

    use crate::utils::assert_eq_tensor;

    const LEVELS: [u8; 4] = [64, 128, 192, 254];

    #[test]
    fn sanity_check_benchmark() {
        let _ = tch::no_grad_guard();
        let rand = Tensor::rand(&[350, 1, 64, 64], (Kind::Float, tch::Device::Cpu));

        for level in LEVELS.iter() {
            let start = std::time::Instant::now();
            let glcm_cpu = super::glcm_cpu(&rand, (1, 0), *level, None, false);
            for _ in 0..10 {
                let _ = super::glcm_cpu(&rand, (1, 0), *level, None, false);
            }
            let t_cpu = start.elapsed().as_millis();
            let _ = super::glcm_gpu(
                &rand.to_device(Device::cuda_if_available()),
                (1, 0),
                *level,
                None,
                false,
            );
            let start = std::time::Instant::now();
            let glcm_gpu = super::glcm_gpu(
                &rand.to_device(Device::cuda_if_available()),
                (1, 0),
                *level,
                None,
                false,
            );
            for _ in 0..10 {
                let _ = super::glcm_gpu(
                    &rand.to_device(Device::cuda_if_available()),
                    (1, 0),
                    *level,
                    None,
                    false,
                );
            }
            let t_gpu = start.elapsed().as_millis();
            println!("{}", f32::from(glcm_cpu.sum(Kind::Float)));
            println!("{}", f32::from(glcm_gpu.sum(Kind::Float)));
            assert_eq_tensor(&glcm_cpu, &glcm_gpu.to(Device::Cpu));
            println!("level: {}, cpu: {}, gpu: {}", level, t_cpu, t_gpu);
        }
    }

    #[test]
    fn test_glcm_no_mask() {
        #[rustfmt::skip]
        let input = Tensor::of_slice(&[
            1.0, 1.0, 2.0, 1.0,
            2.0, 1.0, 1.0, 1.0,
            3.0, 2.0, 3.0, 1.0,
            3.0, 2.0, 1.0, 2.0,             

            1.0, 1.0, 2.0, 1.0,
            2.0, 1.0, 1.0, 1.0,
            3.0, 2.0, 3.0, 1.0,
            3.0, 2.0, 1.0, 2.0
        ]);

        #[rustfmt::skip]
        let expected = Tensor::of_slice(&[
            0.25, 0.17, 0.0,
            0.25, 0.0,  0.08,
            0.08, 0.17, 0.0,
        
            0.25, 0.17, 0.0,
            0.25, 0.0,  0.08,
            0.08, 0.17, 0.0,
        ]).view((2, 3, 3));

        let input = (input.view((2, 1, 4, 4)) - 1.0) / 3.0;
        let rand = Tensor::rand(&[2, 1, 4, 4], (Kind::Float, tch::Device::Cpu));
        let input = Tensor::cat(&[input, rand], 0);
        let glcm = super::glcm(&input, (0, 1), 3, None, false);
        let glcm_gpu = super::glcm_gpu(&input, (0, 1), 3, None, false);
        let glcm_cpu = super::glcm_cpu(&input, (0, 1), 3, None, false);

        let glcm = glcm.i(..2).round_decimals(2);
        let glcm_gpu = glcm_gpu.i(..2).round_decimals(2);
        let glcm_cpu = glcm_cpu.i(..2).round_decimals(2);

        assert_eq_tensor(&glcm_gpu, &expected);
        assert_eq_tensor(&glcm_cpu, &expected);
        assert_eq_tensor(&glcm, &expected);
    }

    #[test]
    fn test_glcm_mask() {
        #[rustfmt::skip]
        let input = Tensor::of_slice(&[
            1.0, 1.0, 2.0, 1.0,
            2.0, 1.0, 1.0, 1.0,
            3.0, 2.0, 3.0, 1.0,
            3.0, 2.0, 1.0, 2.0
        ]);
        let input = (input.view((1, 1, 4, 4)) - 1.0) / 3.0;

        #[rustfmt::skip]
        let mask = Tensor::of_slice(&[
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.0
        ]).view((1, 1, 4, 4));

        #[rustfmt::skip]
        let expected = Tensor::of_slice(&[
            3.0, 1.0, 0.0,
            3.0, 0.0, 1.0,
            1.0, 2.0, 0.0,
        ]).view((1, 3, 3)) / 11.0;

        let glcm = super::glcm(&input, (0, 1), 3, Some(&mask), false);
        let glcm_cpu = super::glcm_cpu(&input, (0, 1), 3, Some(&mask), false);
        let glcm_gpu = super::glcm_gpu(&input, (0, 1), 3, Some(&mask), false);

        assert_eq_tensor(&glcm, &expected);
        assert_eq_tensor(&glcm_cpu, &expected);
        assert_eq_tensor(&glcm_gpu, &expected);
    }

    // TODO: Add test for more complex GLCMs
}
