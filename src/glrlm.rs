/*!
# Gray Level Run Length Matrix (GLRLM).

This module contains generation of GLRLM and features that can be extracted from it.

## Features

 */

use tch::{Tensor, index::*, Kind};


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
    assert!(num_levels >= 2 && num_levels <= 254, "num_levels must be in the range [2, 254]");
    assert!(max_run_length >= 1, "max_run_length must be at least 1");
    assert!(dx.abs() < 2 && dy.abs() < 2, "dx and dy must be in the range [-1, 1]");
    assert!(dx != 0 || dy != 0, "dx and dy cannot both be 0");
    let min = f32::from(image.min());
    let max = f32::from(image.max());
    let kind = image.kind();
    assert!(min >= 0.0 && max <= 1.0 && kind == Kind::Float, "image must be float in the range [0, 1]");

    // map the image to the range [0, num_levels - 1]
    let mut image = image * num_levels as f64;
    // If we use a mask we add 255 to the masked pixels so that they are not counted in the GLCM.
    if let Some(mask) = mask {
        assert!(mask.size() == image.size(), "mask must have the same size as image");
        let mask = mask.to_kind(Kind::Float);
        let mask = (mask - 1.0) * - 255.0;
        image += mask;
    }
    let image = image.clamp(0.0, 255.0).to_kind(tch::Kind::Uint8);
    

    // We create a mask where the pixels are 1 if they aren't the first pixel of a run.
    let mask = {
        let conv = Tensor::zeros(&[3, 3], (tch::Kind::Float, image.device()));
        drop(conv.i((1, 1)).fill_(1.0));
        drop(conv.i((1 - dy, 1 - dx)).fill_(-1.0));
        let conv = conv.view([1, 1, 3, 3]);
        let image = image.to_kind(Kind::Float).conv2d::<&Tensor>(&conv, None, &[1, 1], &[1, 1], &[1, 1], 1);
        
        image.eq(0)
    };


    // Computing the same shade run length 
    let run_length = Tensor::ones_like(&image);
    let mut dest_slice = run_length.i((.., .., dy.max(0)..(height + dy).min(height), dx.max(0)..(width + dx).min(width)));
    let neigh_slice = run_length.i((.., .., (-dy).max(0)..(height - dy).min(height), (-dx).max(0)..(width - dx).min(width)));
    let mask_slice = mask.i((.., .., dy.max(0)..(height + dy).min(height), dx.max(0)..(width + dx).min(width)));
    for i in 0..max_run_length-2{
        dest_slice.copy_(&(&neigh_slice * &mask_slice + 1));
        println!("--------------- i: {} --------------", i);
    }
    
    // Generate a mask that only is true for the furthers point of a gray run 
    let mask = {
        let slice = mask.i((
            ..,
            .., 
            dy.max(0)..((height + dy).min(height)), 
            dx.max(0)..((width + dx).min(width))
        ));
        let slice = slice.pad(&[(-dx).max(0), dx.max(0), (-dy).max(0), dy.max(0)], "constant", Some(0.0)).to_kind(Kind::Int);
        -slice + 1.0
    };

    let run_length = (run_length * mask).clamp(0, max_run_length).to_kind(Kind::Int);
    
    let glrlm = Tensor::zeros(&[batch_size, num_levels as i64, max_run_length], (Kind::Int64, image.device()));
    for level in 0..num_levels{
        let lmask = image.eq(level as i64);
        for len in 1..=max_run_length{
            let rmask = run_length.eq(len);
            let count = (&lmask * rmask).sum_dim_intlist(Some(&[1, 2, 3][..]), false, Kind::Int);
            glrlm.i((.., level as i64, len - 1)).copy_(&count);
        }
    }
    glrlm
}

#[cfg(test)]
mod test {
    use tch::{Tensor, Kind, Device, index::*};
    use crate::{glrlm::glrlm, utils::assert_eq_tensor};

    #[test]
    fn test_glrlm_0deg(){
        let image = Tensor::of_slice(&[
            2.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 3.0, 0.0, 0.0, 2.0, 0.0, 3.0, 3.0,
            2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 3.0, 3.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0,
            3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 2.0, 0.0, 0.0, 3.0, 3.0,
        ]).view([1, 1, 3, 15]).to_kind(Kind::Float);
        let expected = Tensor::of_slice(&[
            4, 3, 1, 1,
            1, 2, 1, 0,
            10, 0, 0, 0,
            3, 2, 1, 0,
        ]).view([1, 4, 4]);
        let image = image / 4.0;
        let glrlm = glrlm(&image, 4, 4, (1, 0), None);
        assert_eq_tensor(&glrlm, &expected);
    }

    #[test]
    fn test_glrlm_45deg(){
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
        let glrlm = glrlm(&(&image/2.0), 2, 4, (1, 1), None);
        let expected = Tensor::of_slice(&[
            15, 12, 1, 0,
            13, 9, 1, 5,
        ]).view([1, 2, 4]);

        assert_eq_tensor(&glrlm, &expected);
    }

    #[test]
    fn test_glrlm_90deg(){
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
        let expect = Tensor::of_slice(&[
            12, 9, 3, 1,
            15, 7, 5, 3
        ]).view([1, 2, 4]);
        let glrlm = glrlm(&(&image/2.0), 2, 4, (0, 1), None);
        assert_eq_tensor(&glrlm, &expect);
    }

    const N: i64 = 100;
    #[test]
    fn sanity_check(){
        let mut image = Tensor::zeros(&[N, 1, 10, 1000], (Kind::Float, Device::Cpu));
        drop(image.uniform_(0.0, 99.9999));
        image/=100.0;
        let glrlm = glrlm(&(&image/2.0), 100, 10, (0, 1), None);
        // Check that the sum of all the elements is equal to the number of pixels
        let i = Tensor::arange(10, (Kind::Int64, Device::Cpu)) + 1;
        let i = i.view([1, -1]).repeat(&[100, 1]);
        assert_eq!(i64::from((glrlm * i).sum(Kind::Int64)), N * 10_000, "The sum of all longest run lengths should be equal to the number of pixels");
    }
}