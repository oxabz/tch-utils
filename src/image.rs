use image;
use tch::Tensor;

pub trait ImageTensorExt {
    fn to_image(&self) -> image::DynamicImage;
    
    fn from_image(image: image::DynamicImage) -> Self;
}

impl ImageTensorExt for Tensor {
    fn to_image(&self) -> image::DynamicImage {
        todo!()
    }

    fn from_image(image: image::DynamicImage) -> Self {
        todo!()
    }
}