/*!
# Gray Level Run Length Matrix (GLRLM).

This module contains generation of GLRLM and features that can be extracted from it.

## Features

 */

use tch::{index::*, Kind, Tensor};

const GLRLM_BINCOUNT_SIZE: i64 = 0x00FF_FFFF;

/**
Generate GLRLM from an image.

# Arguments
- `image`: [N, 1, H, W] grayscale tensor of the image with values in [0, 1].
- `num_levels`: number of gray levels to use. From 2 to 254 with a mask and From 2 to 255 otherwise.
- `max_run_length`: maximum run length to use.
- `direction` : a tuple (dx, dy) of the direction of the run length.
- `normalize`: whether to normalize the GLRLM.
- `mask`: [N, 1, H, W] tensor of the mask with values in [0, 1].

# Returns
- [N, num_levels, max_run_length] GLRLM tensor.
 */
pub fn glrlm(
    image: &Tensor,
    num_levels: u8,
    max_run_length: i64,
    direction: (i64, i64),
    mask: Option<&Tensor>,
) -> Tensor {
    let (batch_size, _, height, width) = image.size4().unwrap();
    let (dx, dy) = direction;
    assert!(
        (2..=254).contains(&num_levels),
        "num_levels must be in the range [2, 254]"
    );
    assert!(max_run_length >= 1, "max_run_length must be at least 1");
    assert!(
        dx.abs() < 2 && dy.abs() < 2,
        "dx and dy must be in the range [-1, 1]"
    );
    assert!(dx != 0 || dy != 0, "dx and dy cannot both be 0");
    let min = f32::from(image.min());
    let max = f32::from(image.max());
    let kind = image.kind();
    assert!(
        min >= 0.0 && max <= 1.0 && kind == Kind::Float,
        "image must be float in the range [0, 1]"
    );

    // map the image to the range [0, num_levels - 1]
    let mut image = image * num_levels as f64;
    // If we use a mask we add 255 to the masked pixels so that they are not counted in the GLCM.
    if let Some(mask) = mask {
        assert!(
            mask.size() == image.size(),
            "mask must have the same size as image"
        );
        image *= mask;
        let mask = mask.to_kind(Kind::Float);
        let mask = (mask - 1.0) * -(num_levels as f64);
        image += mask;
    }
    let image = image.clamp(0.0, 255.0).to_kind(tch::Kind::Uint8);

    // We create a mask where the pixels are 1 if they aren't the first pixel of a run.
    let mask = {
        let conv = Tensor::zeros(&[3, 3], (tch::Kind::Float, image.device()));
        drop(conv.i((1, 1)).fill_(1.0));
        drop(conv.i((1 - dy, 1 - dx)).fill_(-1.0));
        let conv = conv.view([1, 1, 3, 3]);
        let image =
            image
                .to_kind(Kind::Float)
                .conv2d::<&Tensor>(&conv, None, &[1, 1], &[1, 1], &[1, 1], 1);

        image.eq(0)
    };

    // Computing the same shade run length
    let run_length = Tensor::ones_like(&image);
    let mut dest_slice = run_length.i((
        ..,
        ..,
        dy.max(0)..(height + dy).min(height),
        dx.max(0)..(width + dx).min(width),
    ));
    let neigh_slice = run_length.i((
        ..,
        ..,
        (-dy).max(0)..(height - dy).min(height),
        (-dx).max(0)..(width - dx).min(width),
    ));
    let mask_slice = mask.i((
        ..,
        ..,
        dy.max(0)..(height + dy).min(height),
        dx.max(0)..(width + dx).min(width),
    ));
    for _ in 0..max_run_length - 1 {
        dest_slice.copy_(&(&neigh_slice * &mask_slice + 1));
    }

    // Generate a mask that only is true for the furthers point of a gray run
    let mask = {
        let slice = mask.i((
            ..,
            ..,
            dy.max(0)..((height + dy).min(height)),
            dx.max(0)..((width + dx).min(width)),
        ));
        let slice = slice
            .pad(
                &[(-dx).max(0), dx.max(0), (-dy).max(0), dy.max(0)],
                "constant",
                Some(0.0),
            )
            .to_kind(Kind::Int);
        -slice + 1.0
    };

    let run_length = (run_length * mask)
        .clamp(0, max_run_length)
        .to_kind(Kind::Int);

    let num_levels = num_levels as i64 + 1; // We add 1 to account for the masked pixels
    let group_size = ((GLRLM_BINCOUNT_SIZE - 1) / num_levels * max_run_length).min(batch_size);
    let group_count = (batch_size as f64 / group_size as f64).ceil() as i64;
    let group_size = (batch_size as f64 / group_count as f64).ceil() as i64;
    let batch_idx = Tensor::arange(batch_size, (Kind::Int64, image.device())).remainder(group_size);
    let batch_idx = batch_idx.view([-1, 1, 1, 1]);
    let pairs = &run_length
        + image.to_kind(Kind::Int64) * (max_run_length + 1)
        + batch_idx * num_levels * (max_run_length + 1);

    let pairs = pairs.tensor_split(group_count, 0);
    let glrlms = pairs
        .iter()
        .map(|t| (t.size()[0], t))
        .map(|(s, t)| (s, t.view([-1])))
        .map(|(s, t)| {
            (
                s,
                t.bincount::<&Tensor>(None, num_levels * (max_run_length + 1) * s),
            )
        })
        .map(|(s, t)| t.view([s, num_levels, max_run_length + 1]))
        .collect::<Vec<_>>();

    let glrlm = Tensor::cat(&glrlms, 0);
    glrlm.i((.., ..(num_levels - 1), 1..))
}

/**
Contains the feature computation of features that can be extracted from a GLRLM.
 */
pub mod features {
    use tch::{Kind, Tensor};

    use crate::tensor_ext::TensorExt;

    /**
    Contains the features that can be extracted from a GLRLM.
    Each field is a [N] tensor of the corresponding feature for each element of the batch.
     */
    pub struct GlrlmFeatures {
        pub run_percentage: Tensor,
        pub run_length_mean: Tensor,
        pub run_length_variance: Tensor,
        pub gray_level_non_uniformity: Tensor,
        pub run_length_non_uniformity: Tensor,
        // Emphasis features
        pub short_run_emphasis: Tensor,
        pub long_run_emphasis: Tensor,
        pub low_gray_level_run_emphasis: Tensor,
        pub high_gray_level_run_emphasis: Tensor,
        pub short_run_low_gray_level_emphasis: Tensor,
        pub short_run_high_gray_level_emphasis: Tensor,
        pub long_run_low_gray_level_emphasis: Tensor,
        pub long_run_high_gray_level_emphasis: Tensor,
        pub short_run_mid_gray_level_emphasis: Tensor,
        pub long_run_mid_gray_level_emphasis: Tensor,
        pub short_run_extreme_gray_level_emphasis: Tensor,
        pub long_run_extreme_gray_level_emphasis: Tensor,
    }

    /**
    Extract a set of features from a GLRLM tensor.

    # Arguments
    - `glrlm`: [N, num_levels, max_run_length] GLRLM tensor.
    - `pixel_count`: [N] tensor of the number of pixels in each image.
        Optional. If not provided it will be computed from the GLRLM. Which is accurate if the GLRLM's max_run_length is above the effective max run length.
        This only affects the computation of the run percentage feature.

    # Returns
    A [GlrlmFeatures] struct containing the features.
    */
    pub fn glrlm_features(glrlm: &Tensor, pixel_count: Option<&Tensor>) -> GlrlmFeatures {
        // Normalize the GLRLM
        let nruns = glrlm.sum_kdims([-1, -2]);
        let glrlm = glrlm / &nruns;
        let tensor_option = (glrlm.kind(), glrlm.device());

        let (_batch_size, num_levels, max_run_length) = glrlm.size3().unwrap();

        let mut run_lengths = Tensor::arange(max_run_length, tensor_option);
        run_lengths += 1;
        let run_lengths = run_lengths.unsqueeze(0);
        let run_lengths2 = run_lengths.square();

        let gray_levels = Tensor::arange(num_levels, tensor_option);
        let gray_levels = gray_levels.unsqueeze(0);
        let gray_levels2 = gray_levels.square();

        // Gray Level Run-Length Vector [N, GLRLM_LEVELS]
        let glrlv = glrlm.sum_dim(-1);

        // Run-length Run Number Vector [N, GLRLM_MAX_LENGTH]
        let rlrnv = glrlm.sum_dim(-2);

        // Run Percentage [N], Run Length Mean [N] & Run Length Variance [N]
        let (mut run_percentage, run_length_mean, run_length_variance) = {
            let mut jrlnv = &rlrnv * run_lengths;

            let run_percentage = 1.0 / jrlnv.sum_dim(-1);
            let run_length_mean = jrlnv.mean_dim(Some(&[-1][..]), false, Kind::Float);
            jrlnv -= &run_length_mean;
            let jrlnv = jrlnv.square_();
            let run_length_variance = jrlnv.mean_dim(Some(&[-1][..]), false, Kind::Float);

            (run_percentage, run_length_mean, run_length_variance)
        };
        if let Some(pixel_count) = pixel_count {
            run_percentage = nruns.squeeze() / pixel_count.to_kind(Kind::Float);
        }

        // Gray Level Non-Uniformity [N]
        let gray_level_non_uniformity = glrlv.square().sum_dim(-1);

        // Run Length Non-Uniformity [N]
        let run_length_non_uniformity = rlrnv.square().sum_dim(-1);

        // Emphasis features

        // Short Run Emphasis [N]
        let short_run_emphasis = (&rlrnv * &run_lengths2).sum_dim(-1);

        // Long Run Emphasis [N]
        let long_run_emphasis = (&rlrnv / &run_lengths2).sum_dim(-1);

        // Low Gray Level Run Emphasis [N]
        let low_gray_level_run_emphasis = (&glrlv / &gray_levels2).sum_dim(-1);

        // High Gray Level Run Emphasis [N]
        let high_gray_level_run_emphasis = (&glrlv * &gray_levels2).sum_dim(-1);

        // We use this copy to compute the next features to avoid to duplicate memory & computation
        let mut weighted_glrlm = glrlm.copy();

        // Short & Long Run High Gray Level Emphasis [N]
        let (short_run_high_gray_level_emphasis, long_run_high_gray_level_emphasis) = {
            weighted_glrlm *= gray_levels2.unsqueeze(-1); // We weight the GLRLM by the square of the gray levels

            let short_run_high_gray_level_emphasis =
                (&weighted_glrlm / run_lengths2.unsqueeze(-2)).sum_dims([-1, -2]);
            let long_run_high_gray_level_emphasis =
                (&weighted_glrlm * run_lengths2.unsqueeze(-2)).sum_dims([-1, -2]);

            (
                short_run_high_gray_level_emphasis,
                long_run_high_gray_level_emphasis,
            )
        };

        // Short & Long Run Low Gray Level Emphasis [N]
        let (short_run_low_gray_level_emphasis, long_run_low_gray_level_emphasis) = {
            weighted_glrlm.copy_(&glrlm); // We reset the weighted GLRLM
            weighted_glrlm /= gray_levels2.unsqueeze(-1); // We weight the GLRLM by the inverse of the square of the gray levels

            let short_run_low_gray_level_emphasis =
                (&glrlm / run_lengths2.unsqueeze(-2)).sum_dims([-1, -2]);
            let long_run_low_gray_level_emphasis =
                (&glrlm * run_lengths2.unsqueeze(-2)).sum_dims([-1, -2]);

            (
                short_run_low_gray_level_emphasis,
                long_run_low_gray_level_emphasis,
            )
        };

        // Short & Long Run Mid Gray Level Emphasis [N]
        let (short_run_mid_gray_level_emphasis, long_run_mid_gray_level_emphasis) = {
            weighted_glrlm.copy_(&glrlm); // We reset the weighted GLRLM
            let i = Tensor::arange(num_levels, tensor_option) - (num_levels - 1) / 2;
            let i = i.unsqueeze(0).unsqueeze(-1).square_();
            weighted_glrlm /= i.unsqueeze(-1);

            let short_run_mid_gray_level_emphasis =
                (&glrlm / run_lengths2.unsqueeze(-2)).sum_dims([-1, -2]);
            let long_run_mid_gray_level_emphasis =
                (&glrlm * run_lengths2.unsqueeze(-2)).sum_dims([-1, -2]);

            (
                short_run_mid_gray_level_emphasis,
                long_run_mid_gray_level_emphasis,
            )
        };

        // Short & Long Run Extreme Gray Level Emphasis [N]
        let (short_run_extreme_gray_level_emphasis, long_run_extreme_gray_level_emphasis) = {
            weighted_glrlm.copy_(&glrlm); // We reset the weighted GLRLM
            let j = Tensor::arange(num_levels, tensor_option) - (num_levels - 1) / 2;
            let j = j.unsqueeze(0).unsqueeze(-1).square_();
            weighted_glrlm *= j.unsqueeze(-1);

            let short_run_extreme_gray_level_emphasis =
                (&glrlm / run_lengths2.unsqueeze(-2)).sum_dims([-1, -2]);
            let long_run_extreme_gray_level_emphasis =
                (&glrlm * run_lengths2.unsqueeze(-2)).sum_dims([-1, -2]);

            (
                short_run_extreme_gray_level_emphasis,
                long_run_extreme_gray_level_emphasis,
            )
        };

        GlrlmFeatures {
            run_percentage,
            run_length_mean,
            run_length_variance,
            gray_level_non_uniformity,
            run_length_non_uniformity,
            short_run_emphasis,
            long_run_emphasis,
            low_gray_level_run_emphasis,
            high_gray_level_run_emphasis,
            short_run_low_gray_level_emphasis,
            short_run_high_gray_level_emphasis,
            long_run_low_gray_level_emphasis,
            long_run_high_gray_level_emphasis,
            short_run_mid_gray_level_emphasis,
            long_run_mid_gray_level_emphasis,
            short_run_extreme_gray_level_emphasis,
            long_run_extreme_gray_level_emphasis,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{glrlm::glrlm, utils::assert_eq_tensor};
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_glrlm_0deg() {
        #[rustfmt::skip]
        let image = Tensor::of_slice(&[
            2.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 3.0, 0.0, 0.0, 2.0, 0.0, 3.0, 3.0,
            2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 3.0, 3.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0,
            3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 2.0, 0.0, 0.0, 3.0, 3.0,
        ]).view([1, 1, 3, 15]).to_kind(Kind::Float).repeat(&[500, 1, 1, 1]);

        #[rustfmt::skip]
        let expected = Tensor::of_slice(&[
            4, 3, 1, 1,
            1, 2, 1, 0,
            10, 0, 0, 0,
            3, 2, 1, 0,
        ]).view([1, 4, 4]).repeat(&[500, 1, 1]);
        let image = image / 4.0;
        let glrlm = glrlm(&image, 4, 4, (1, 0), None);
        assert_eq_tensor(&glrlm, &expected);
    }

    #[test]
    fn test_glrlm_45deg() {
        #[rustfmt::skip]
        let image = Tensor::of_slice(&[
            1, 0, 1, 0, 0, 1, 1, 1, 0, 1,
            1, 0, 1, 1, 1, 1, 1, 0, 1, 0,
            1, 0, 1, 1, 0, 1, 1, 0, 1, 1,
            1, 0, 0, 1, 1, 0, 1, 1, 1, 0,
            0, 1, 0, 1, 1, 1, 0, 1, 1, 1,
            1, 0, 0, 1, 1, 1, 1, 1, 1, 0,
            1, 0, 1, 0, 0, 0, 0, 1, 0, 1,
            0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
            0, 1, 0, 1, 1, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 1, 0, 0, 1, 0]).to_kind(Kind::Float).view([1, 1, 10, 10]);
        let glrlm = glrlm(&(&image / 2.0), 2, 4, (1, 1), None);
        let expected = Tensor::of_slice(&[15, 12, 1, 0, 13, 9, 1, 5]).view([1, 2, 4]);

        assert_eq_tensor(&glrlm, &expected);
    }

    #[test]
    fn test_glrlm_90deg() {
        #[rustfmt::skip]
        let image = Tensor::of_slice(&[
            0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 1, 1, 1,
            1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
            1, 1, 1, 0, 1, 0, 1, 1, 0, 0,
            1, 1, 1, 0, 0, 1, 1, 1, 0, 1,
            0, 1, 1, 1, 0, 1, 1, 1, 1, 1,
            0, 1, 0, 1, 1, 1, 1, 0, 1, 1,
            1, 0, 0, 0, 0, 0, 1, 1, 0, 0,
            1, 0, 0, 1, 1, 0, 0, 0, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
        ]).view([1, 1, 10, 10]);

        #[rustfmt::skip]
        let expect = Tensor::of_slice(&[
            12, 9, 3, 1,
            15, 7, 5, 3
        ]).view([1, 2, 4]);
        let glrlm = glrlm(&(&image / 2.0), 2, 4, (0, 1), None);
        assert_eq_tensor(&glrlm, &expect);
    }

    const N: i64 = 100;
    #[test]
    fn sanity_check() {
        let mut image = Tensor::zeros(&[N, 1, 10, 1_000], (Kind::Float, Device::Cpu));
        drop(image.uniform_(0.0, 99.9999));
        image /= 100.0;
        let glrlm = glrlm(&image, 100, 10, (0, 1), None);
        // Check that the sum of all the elements is equal to the number of pixels
        let i = Tensor::arange(10, (Kind::Int64, Device::Cpu)) + 1;
        let i = i.view([1, 1, -1]);
        assert_eq!(
            i64::from((glrlm * i).sum(Kind::Int64)),
            N * 10_000,
            "The sum of all longest run lengths should be equal to the number of pixels"
        );
    }
}
