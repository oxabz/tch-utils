use tch::{Kind, Tensor, index::*};

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

# Returns
- Tensor of shape [N, num_shades, num_shades] where N is the batch size.
    The values are in the range [0, 1] and represent the normalized GLCM.
    dim 1 is the reference shade and dim 2 is the neighbor shade.

# Big O
- Time: O(num_shades²) if we have N * W * H threads.

 */
pub fn glcm(image: &Tensor, offset: (i64, i64), num_shades: u8, mask: Option<&Tensor>) -> Tensor {
    let (offset_y, offset_x) = offset;
    let (offset_y, offset_x) = (offset_y as i64, offset_x as i64);
    let (batch_size, _, height, width) = image.size4().unwrap();

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
        if offset_y >= 0 { 0..(height-offset_y) } else { offset_y..height },
        if offset_x >= 0 { 0..(width-offset_x) } else { offset_x..width },
    );
    let nslice = (
        ..,
        ..,
        if offset_y >= 0 { offset_y..height } else { 0..(height-offset_y) },
        if offset_x >= 0 { offset_x..width } else { 0..-(width-offset_x) },
    );    
    
    let glcm = Tensor::zeros(
        &[batch_size, num_shades as i64, num_shades as i64],
        (Kind::Float, image.device()),
    );
    for reference_shade in 0..num_shades {
        let reference_mask = image.eq(reference_shade as i64);
        for neighbor_shade in 0..num_shades {
            let neighbor_mask = image.eq(neighbor_shade as i64); 
            let mut slice = glcm.i((.., reference_shade as i64, neighbor_shade as i64));
            slice += (reference_mask.i(rslice.clone()) * neighbor_mask.i(nslice.clone())).sum_dim_intlist(Some(&[1, 2, 3][..]), false, Kind::Float);
        }
    }
    &glcm / glcm.sum_dim_intlist(Some(&[1, 2][..]), false, Kind::Float)
}

#[cfg(test)]
mod test {
    use tch::Tensor;

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
        let glcm = super::glcm(&input, (1, 0), 3, None);

        assert_eq_tensor(&glcm, &expected);
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
        
        let glcm = super::glcm(&input, (1, 0), 3, Some(&mask));
        
        assert_eq_tensor(&glcm, &expected);

    }

    // TODO: Add test for more complex GLCMs
}