/*!
Adds a trait to convert between `tch::Tensor` and `image::DynamicImage`.
 */

use image;
use tch::{Tensor, Kind};

/**
Trait to convert between `tch::Tensor` and `image::DynamicImage`.
 */
pub trait ImageTensorExt {
    /**
    Converts a `tch::Tensor` to a `image::DynamicImage`.
    # Arguments
    - `self` - The [C, H, W] `tch::Tensor` to convert to an Image.
        with C = 4 (RGBA), C = 3 (RGB), C = 2 (RGB) or C = 1 (RGB).
        we expect a non complex tensor.
        if every type will be converted to `f32` except `u8`.

    # Returns
    - The `image::DynamicImage` representation of the `tch::Tensor`.
     */
    fn to_image(&self) -> image::DynamicImage;
    
    /**
    Converts a `image::DynamicImage` to a `tch::Tensor`.

    # Arguments
    - `image` - The `image::DynamicImage` to convert to a `tch::Tensor`.

    # Returns
    - The [C, H, W] `tch::Tensor` representation of the `image::DynamicImage`.
    
     */
    fn from_image(image: image::DynamicImage) -> Self;
}

impl ImageTensorExt for Tensor {
    fn to_image(&self) -> image::DynamicImage {
        let size = self.size();
        let kind = self.kind();
        assert!(size.len() == 3, "Tensor must be of shape [C, H, W] (got {:?})", size);
        let [channels, height, width] = size[0..3] else { unreachable!()};
        assert!((1..=4).contains(&channels), "Tensor must have 4, 3, 2 or 1 channels (got {:?})", channels);
        assert!(kind != Kind::ComplexFloat && kind != Kind::ComplexDouble, "Tensor must be non complex (got {:?})", kind);
        
        let tensor = match channels {
            3..=4 => self.shallow_clone(),
            2 => {
                let z = Tensor::zeros(&[1, height, width], (tch::Kind::Float, self.device()));
                Tensor::cat(&[&z, self], 0)
            },
            1 => self.repeat(&[3, 1, 1]),
            _ => unreachable!(),
        };
        let tensor = tensor.permute(&[2, 1, 0]);
        match (channels, kind) {
            (1..=3, Kind::Uint8)=>{
                let data = Vec::<u8>::from(tensor);
                image::DynamicImage::ImageRgb8(image::ImageBuffer::from_raw(width as u32, height as u32, data).unwrap())
            },
            (4, Kind::Uint8)=>{
                let data = Vec::<u8>::from(tensor);
                image::DynamicImage::ImageRgba8(image::ImageBuffer::from_raw(width as u32, height as u32, data).unwrap())
            },
            (1..=3, _)=>{
                let tensor = tensor.to_kind(Kind::Float);
                let data = Vec::<f32>::from(tensor);
                image::DynamicImage::ImageRgb32F(image::ImageBuffer::from_raw(width as u32, height as u32, data).unwrap())
            },
            (4, _)=>{
                let tensor = tensor.to_kind(Kind::Float);
                let data = Vec::<f32>::from(tensor);
                image::DynamicImage::ImageRgba32F(image::ImageBuffer::from_raw(width as u32, height as u32, data).unwrap())
            },
            _ => unreachable!(),
        }
    }

    fn from_image(image: image::DynamicImage) -> Self {
        let (width, height) = (image.width(), image.height());
        let image = image.to_rgba32f();
        let data = image.into_vec();
        let tensor = Tensor::of_slice(&data);
        tensor.reshape(&[width as i64, height as i64, 4]).permute(&[2, 1, 0])
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::assert_eq_tensor;

    use super::*;
    use tch::Tensor;

    #[test]
    fn test_image_tensor() {
        let image = image::open("test-assets/convert/basic.png").unwrap();
        let tensor = Tensor::from_image(image.clone());
        let image2 = tensor.to_image();
        let tensor2 = Tensor::from_image(image2.clone());
        assert_eq_tensor(&tensor, &tensor2);
        
        
        let image = image::open("test-assets/convert/cat.jpg").unwrap();
        let tensor = Tensor::from_image(image.clone());
        let image2 = tensor.to_image();
        image2.to_rgb8().save("/tmp/cat.jpg").unwrap();
        let tensor2 = Tensor::from_image(image2.clone());
        assert_eq_tensor(&tensor, &tensor2);
    }
}