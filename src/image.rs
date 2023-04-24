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
    
    fn from_image(image: image::DynamicImage) -> Self;
}

impl ImageTensorExt for Tensor {
    fn to_image(&self) -> image::DynamicImage {
        let size = self.size();
        let kind = self.kind();
        assert!(size.len() == 3, "Tensor must be of shape [C, H, W] (got {:?})", size);
        let [channels, height, width] = size[0..3] else { unreachable!()};
        assert!(channels <= 4 && channels >= 1, "Tensor must have 4, 3, 2 or 1 channels (got {:?})", channels);
        assert!(kind != Kind::ComplexFloat && kind != Kind::ComplexDouble, "Tensor must be non complex (got {:?})", kind);
        
        let tensor = match channels {
            4 | 3 => self.shallow_clone(),
            2 => {
                let z = Tensor::zeros(&[1, height, width], (tch::Kind::Float, self.device()));
                Tensor::cat(&[&z, &self], 0)
            },
            1 => self.repeat(&[3, 1, 1]),
            _ => unreachable!(),
        };
        let tensor = tensor.permute(&[1, 2, 0]);
        match (channels, kind) {
            (1|2|3, Kind::Uint8)=>{
                let data = Vec::<u8>::from(tensor);
                image::DynamicImage::ImageRgb8(image::ImageBuffer::from_raw(width as u32, height as u32, data).unwrap())
            },
            (4, Kind::Uint8)=>{
                let data = Vec::<u8>::from(tensor);
                image::DynamicImage::ImageRgba8(image::ImageBuffer::from_raw(width as u32, height as u32, data).unwrap())
            },
            (1|2|3, _)=>{
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
        tensor.reshape(&[4, height as i64, width as i64])
    }
}