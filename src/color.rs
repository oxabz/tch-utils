use tch::Tensor;

#[rustfmt::skip]
const HED_FROM_RGB: &[f32; 9] = &[
     1.877_982_7,   -1.007_678_6,  -0.556_115_8,
    -0.065_908_06,   1.134_730_3,  -0.135_521_8,
    -0.601_907_4,   -0.480_414_18,  1.573_588
];

/**
Convert a tensor of RGB values to HED values.

# Arguments
- rgb: Tensor - The tensor of RGB values [N, 3, H, W] with float values in the range [0, 1]

# Returns
Tensor - The tensor of HED values [N, 3, H, W] with float values in the range [0, 1]
 */
pub fn hed_from_rgb(rgb: &Tensor) -> Tensor {
    let hed_from_rgb = Tensor::of_slice(HED_FROM_RGB)
        .view([3, 3])
        .to_device(rgb.device());
    let mut rgb = rgb.clamp_min(1e-6).log_();
    rgb /= (1e-6f64).ln();
    rgb.transpose_(1, 3)
        .matmul(&hed_from_rgb)
        .transpose_(1, 3)
        .clamp_min_(0.0)
}

/**
Convert a tensor of RGB values to HSV values.

# Arguments
- rgb: Tensor - The tensor of RGB values [N, 3, H, W] with float values in the range [0, 1]

# Returns
Tensor - The tensor of HSV values [N, 3, H, W] with float values in the range [0, 360]
 */
pub fn hsv_from_rgb(rgb: &Tensor) -> Tensor {
    let (max /* [N, H, W] */, _) = rgb.max_dim(1, false);
    let (min /* [N, H, W] */, _) = rgb.min_dim(1, false);
    let delta /* [N, H, W] */ = &max - min;
    let mut h  /* [N, H, W] */ = Tensor::zeros_like(&delta);
    h += rgb.select(1, 0).eq_tensor(&max).to_kind(tch::Kind::Float)
        * ((rgb.select(1, 1) - rgb.select(1, 2)) / (&delta)).fmod(6.0)
        * 60.0;
    h += rgb.select(1, 1).eq_tensor(&max).to_kind(tch::Kind::Float)
        * ((rgb.select(1, 2) - rgb.select(1, 0)) / (&delta) + 2.0)
        * 60.0;
    h += rgb.select(1, 2).eq_tensor(&max).to_kind(tch::Kind::Float)
        * ((rgb.select(1, 0) - rgb.select(1, 1)) / (&delta) + 4.0)
        * 60.0;
    h = h.where_scalarother(&delta.not_equal(0), 0.0);
    h += 360.0;
    let _ = h.fmod_(360.0);
    let s  /* [N, H, W] */ = &delta / &max;
    let s = s.where_scalarother(&max.not_equal(0), 0);
    let v  /* [N, H, W] */ = max;

    Tensor::stack(&[h, s, v], 1)
}

#[cfg(test)]
mod test {
    /*!
     * Tests for the color module
     * To test the color we use tensors generated by the skimage library in python.
     * The tensor are saved in the `test-assets/colors` folder and loaded in the tests.
     */

    use super::*;
    use crate::utils::{self, assert_tensor_asset_d};
    use tch::index::*;

    #[test]
    fn test_hed_from_rgb() {
        let rgb = utils::dirty_load("test-assets/colors/example.jpg")
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            / 255.0;
        let hed = hed_from_rgb(&rgb);
        let h = hed.i((.., 0)).clamp_(0.0, 1.0) * 255;
        let e = hed.i((.., 1)).clamp_(0.0, 1.0) * 255;
        let d = hed.i((.., 2)).clamp_(0.0, 1.0) * 255;

        tch::vision::image::save(&h, "test-assets/colors/example-h.png")
            .expect("Failed to save asset");
        tch::vision::image::save(&e, "test-assets/colors/example-e.png")
            .expect("Failed to save asset");
        tch::vision::image::save(&d, "test-assets/colors/example-d.png")
            .expect("Failed to save asset");
    }

    #[test]
    fn test_hsv_from_rgb() {
        let rgb = utils::dirty_load("test-assets/colors/original.npy");
        let hsv = hsv_from_rgb(&rgb);
        let mut h = hsv.i((.., 0));
        h /= 360.0;

        assert_tensor_asset_d(&hsv, "test-assets/colors/hsv.npy", 0.002);
    }
}
