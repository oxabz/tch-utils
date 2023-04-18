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
        let (width, height) = (image.width(), image.height());
        let image = image.to_rgba32f();
        let data = image.into_vec();
        let tensor = Tensor::of_slice(&data);
        tensor.reshape(&[4, height as i64, width as i64])
    }
}