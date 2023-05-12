use tch::{Kind, Tensor, index::*};

use crate::tensor_ext::TensorExt;

/**
Computes the gray-level co-occurrence matrix (GLCM) of an image.

The GLCM is a 2D histogram of the co-occurrence of pixel values at a given offset over an image.

# Arguments
- image: Tensor of shape [N, 1, H, W] where N is the batch size, H and W are the height and width of the image.
    The values are expected to be in the range [0, 1].
- offset: The offset of the co-occurrence. The first element is the vertical offset and the second element is the horizontal offset.
- num_shades: The number of shades of gray to use for the GLCM.
    The values are expected to be in the range [2, 254], if a mask is used and [2, 255] if not.
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
pub fn glcm(image: &Tensor, offset: (i64, i64), num_shades: u8, mask: Option<&Tensor>, symmetric: bool) -> Tensor {
    if image.device().is_cuda(){
        glcm_gpu(image, offset, num_shades, mask, symmetric)
    } else {
        glcm_cpu(image, offset, num_shades, mask, symmetric)
    }
}

pub fn glcm_gpu(image: &Tensor, offset: (i64, i64), num_shades: u8, mask: Option<&Tensor>, symmetric: bool) -> Tensor {
    let (offset_y, offset_x) = offset;
    let (offset_y, offset_x) = (offset_y as i64, offset_x as i64);
    let (_, _, height, width) = image.size4().unwrap();

    let mut image = image * num_shades as f64;
    // If we use a mask we add 255 to the masked pixels so that they are not counted in the GLCM.
    if let Some(mask) = mask {
        let mask = (mask - 1.0) * -255.0;
        image += mask;
    }
    let image = image.clamp(0.0, 255.0).to_kind(Kind::Uint8);

    // preping the slices for the image
    let rslice = (
        ..,
        ..,
        (-offset_y).max(0)..(height-offset_y).min(height),
        (-offset_x).max(0)..(width-offset_x).min(width),
    );
    let nslice = (
        ..,
        ..,
        offset_y.max(0)..(height+offset_y).min(height),
        offset_x.max(0)..(width+offset_x).min(width),
    );    
    
    let ref_img = image.i(rslice.clone());
    let neigh_img = image.i(nslice.clone());
    let shades = Tensor::arange(num_shades as i64, (Kind::Float, image.device())).view([1, num_shades as i64, 1, 1]);
    let ref_img = ref_img.eq_tensor(&shades).unsqueeze(2); // [N, S, 1, H, W]
    let neigh_img = neigh_img.eq_tensor(&shades).unsqueeze(1); // [N, 1, S, H, W]
    let mut glcm = (ref_img * neigh_img).sum_dims([-1, -2]); // [N, S, S]

    if symmetric {
        glcm+=glcm.copy().transpose(1, 2);
    }

    let len = glcm.sum_dim_intlist(Some(&[1, 2][..]), false, Kind::Float);
    let len = len.view([-1, 1, 1]);
    &glcm / len
}

pub fn glcm_cpu(image: &Tensor, offset: (i64, i64), num_shades: u8, mask: Option<&Tensor>, symmetric: bool) -> Tensor{
    let (offset_y, offset_x) = offset;

    let (batch_size, _, height, width) = image.size4().unwrap();
    let (batch_size, height, width) = (batch_size as usize, height as usize, width as usize);

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
    let it =((-offset_y).max(0)..(height as i64-offset_y).min(height as i64))
        .flat_map(|y| ((-offset_x).max(0)..(width as i64-offset_x).min(width as i64))
            .map(move|x| (y as usize, x as usize))
        );

    let glcm = (0..batch_size)
        .map(|batch|{
            let mut glcm = vec![0; num_shades as usize * num_shades as usize];
            for (y, x) in it.clone(){
                let reference_shade = image[batch * batch_span + y * width + x] as usize;
                let neighbor_shade = image[batch * batch_span + (y as i64 + offset_y) as usize * width + (x as i64 + offset_x) as usize] as usize;
                if reference_shade >= num_shades as usize || neighbor_shade >= num_shades as usize{
                    continue;
                }
                glcm[reference_shade * num_shades as usize + neighbor_shade] += 1;
            }
            Tensor::of_slice(&glcm).view([num_shades as i64, num_shades as i64])
        });
    let mut glcm = Tensor::stack(&glcm.collect::<Vec<_>>()[..], 0).to_device(device);
    if symmetric {
        glcm+=glcm.copy().transpose(-1, -2);
    }
    let len = glcm.sum_dim_intlist(Some(&[1, 2][..]), false, Kind::Float);
    let len = len.view([-1, 1, 1]);
    glcm / len
}

#[cfg(test)]
mod test {
    use tch::{Tensor};

    use crate::utils::assert_eq_tensor;

    #[test]
    fn test_glcm_no_mask(){
        let input = Tensor::of_slice(&[
            1.0, 1.0, 2.0, 1.0,
            2.0, 1.0, 1.0, 1.0,
            3.0, 2.0, 3.0, 1.0,
            3.0, 2.0, 1.0, 2.0
        ]);

        let expected = Tensor::of_slice(&[
            0.25, 0.17, 0.0,
            0.25, 0.0,  0.08,
            0.08, 0.17, 0.0,
        ]).view((1, 3, 3));
        
        let input = (input.view((1, 1, 4, 4))-1.0) / 3.0;
        let glcm = super::glcm(&input, (1, 0), 3, None, false);
        let glcm_gpu = super::glcm_gpu(&input, (1, 0), 3, None, false);
        let glcm_cpu = super::glcm_cpu(&input, (1, 0), 3, None, false);

        assert_eq_tensor(&glcm, &expected);
        assert_eq_tensor(&glcm_gpu, &expected);
        assert_eq_tensor(&glcm_cpu, &expected);
    }

    #[test]
    fn test_glcm_mask(){
        let input = Tensor::of_slice(&[
            1.0, 1.0, 2.0, 1.0,
            2.0, 1.0, 1.0, 1.0,
            3.0, 2.0, 3.0, 1.0,
            3.0, 2.0, 1.0, 2.0
        ]);
        let input = (input.view((1, 1, 4, 4))-1.0) / 3.0;

        let mask = Tensor::of_slice(&[
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.0
        ]).view((1, 1, 4, 4));

        let expected = Tensor::of_slice(&[
            3.0, 1.0, 0.0,
            3.0, 0.0, 1.0,
            1.0, 2.0, 0.0,
        ]).view((1, 3, 3)) / 11.0;
        
        let glcm = super::glcm(&input, (1, 0), 3, Some(&mask), false);
        
        assert_eq_tensor(&glcm, &expected);

    }

    // TODO: Add test for more complex GLCMs
}