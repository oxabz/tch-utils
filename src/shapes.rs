/*!
 * This module contains functions to generate shapes.
 */

use tch::{Kind, Device, Tensor};

/**
 * Generate an ellipse distance field.
 * 
 * # Arguments
 * - width: i64 - The width of the image
 * - height: i64 - The height of the image
 * - center: (f64, f64) - The center of the ellipse
 * - radii: (f64, f64) - The radii of the ellipse
 * - angle: f64 - The angle of the ellipse
 * 
 * # Returns
 * Tensor - The ellipse distance field [1, H, W] tensor with values in [0, 1]
 */
pub fn ellipse_distance_field(
    width: usize,
    height: usize,
    center: (f64, f64),
    radii: (f64, f64),
    angle: f64,
    device: Device,
) -> Tensor {
    // Create a tensor with the position of each pixel
    let pos = crate::tensor_init::position_tensor_2d((width, height), 1, (Kind::Float, device));
    // Scale the position tensor to the size of the tensor
    let pos = crate::ops_2d::scale_2d(&pos, &[(width/2) as f64, (height/2) as f64]);
    // Translate the position tensor to the center of the ellipse
    let pos = crate::ops_2d::translate_2d(&pos, &[-center.1, -center.0]);
    // Rotate the position tensor to the angle of the ellipse
    let pos = crate::ops_2d::rotate_2d(&pos, angle);
    // Scale the position tensor to the size of the ellipse
    let pos = crate::ops_2d::scale_2d(&pos, &[1.0/radii.1, 1.0/radii.0]);

    (crate::ops_2d::norm_2d(&pos) - 1.0).view([1, height as i64, width as i64])
}

/**
 * Generate an ellipse.
 * 
 * # Arguments
 * - width: usize - The width of the image
 * - height: usize - The height of the image
 * - center: (f64, f64) - The center of the ellipse
 * - radii: (f64, f64) - The radii of the ellipse
 * - angle: f64 - The angle of the ellipse
 * 
 * # Returns
 * Tensor - The ellipse [1, H, W] tensor with value resulting from casting a boolean to the given kind
 */
pub fn ellipse(
    width: usize,
    height: usize,
    center: (f64, f64),
    radii: (f64, f64),
    angle: f64,
    options: (Kind, Device),
) -> Tensor {
    let (kind, device) = options;
    let df = ellipse_distance_field(width, height, center, radii, angle, device);
    (df.lt(0.0)).to_kind(kind)
}

#[cfg(test)]
mod test{
    use super::*;
    use tch::{Device, Kind};

    #[test]
    fn test_ellipse() {
        let centered_circle = ellipse(100, 100, (0.0, 0.0), (25.0, 25.0), 0.0, (Kind::Float, Device::Cpu));
        let offset_circle = ellipse(100, 100, (20.0, 25.0), (20.0, 20.0), 0.0, (Kind::Float, Device::Cpu));
        let ellipse_ = ellipse(100, 100, (0.0, 0.0), (25.0, 10.0), 0.0, (Kind::Float, Device::Cpu));
        let rotated_ellipse = ellipse(100, 100, (0.0, 0.0), (25.0, 10.0), 45.0f64.to_radians(), (Kind::Float, Device::Cpu));
        let centered_rotated_ellipse = ellipse(100, 100, (-20.0, 0.0), (25.0, 10.0), 45.0f64.to_radians(), (Kind::Float, Device::Cpu));

        tch::vision::image::save(&(&centered_circle * 255), "test-results/centered_circle.png").unwrap();
        tch::vision::image::save(&(&offset_circle * 255), "test-results/offset_circle.png").unwrap();
        tch::vision::image::save(&(&ellipse_ * 255), "test-results/ellipse.png").unwrap();
        tch::vision::image::save(&(&rotated_ellipse * 255), "test-results/rotated_ellipse.png").unwrap();
        tch::vision::image::save(&(&centered_rotated_ellipse * 255), "test-results/centered_rotated_ellipse.png").unwrap();

        assert!(f64::from((Tensor::load("assert-assets/centered_circle.pt").unwrap() - centered_circle ).abs().max()) < 1e-6);
        assert!(f64::from((Tensor::load("assert-assets/offset_circle.pt").unwrap() - offset_circle).abs().max()) < 1e-6);
        assert!(f64::from((Tensor::load("assert-assets/ellipse.pt").unwrap() - ellipse_).abs().max()) < 1e-6);
        assert!(f64::from((Tensor::load("assert-assets/rotated_ellipse.pt").unwrap() - rotated_ellipse).abs().max()) < 1e-6);
        assert!(f64::from((Tensor::load("assert-assets/centered_rotated_ellipse.pt").unwrap() - centered_rotated_ellipse).abs().max()) < 1e-6);
    }

    #[test]
    fn test_ellipse_distance_field() {
        let df = ellipse_distance_field(100, 100, (0.0, 0.0), (1.0, 1.0), 0.0, Device::Cpu);       
        tch::vision::image::save(&df.unsqueeze(0), "test-results/ellipse_df.png").unwrap();
    }
}
