use tch::{Kind, Tensor, index::*};

const GLCM_BINCOUNT_SIZE: i64 = 0x00FF_FFFF;

/**
Computes the gray-level co-occurrence matrix (GLCM) of an image.

The GLCM is a 2D histogram of the co-occurrence of pixel values at a given offset over an image.

# Arguments
- image: Tensor of shape [N, 1, H, W] where N is the batch size, H and W are the height and width of the image.
    The values are expected to be in the range [0, 1].
- offset: The offset of the co-occurrence. The first element is the vertical offset and the second element is the horizontal offset.
- num_shades: The number of shades of gray to use for the GLCM.
    The values are expected to be in the range [2, 254]
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

pub fn glcm_gpu(image: &Tensor, offset: (i64, i64), num_shades: u8, mask: Option<&Tensor>, symmetric: bool) -> Tensor{
    let (offset_y, offset_x) = offset;
    let (offset_y, offset_x) = (offset_y as i64, offset_x as i64);
    let (batch_size, _, height, width) = image.size4().unwrap();

    let mut image = image * (num_shades as f64 - 1e-6);
    if let Some(mask) = mask {
        let mask = (mask - 1.0) * -(num_shades as f64);
        image += mask;
    }
    let image = image.floor().clamp(0.0, 255.0).to_kind(Kind::Uint8);

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
    
    // We take a shifted version of the image and compute the GLCM for each pixel.
    let ref_img = image.i(rslice.clone()).to_kind(Kind::Int64);
    let neigh_img = image.i(nslice.clone()).to_kind(Kind::Int64);

    // We then encode the values into a single number.
    // Saddly libtorch doesn't support uint32 tensors so we have to use int32.
    // We map the neighbor's value to the first 8 bits and the reference's value to the next 8 bits.
    // We then use an other 8 bits to encode the batch index we coulduse 15 bit for the batch but it would result in a gigabyte sized tensor.
    let num_shades = (num_shades + 1) as i64; // Number of shades + 1 for the masked pixels
    let group_size = ((GLCM_BINCOUNT_SIZE-1) / num_shades.pow(2)).min(batch_size);
    println!("group_size: {}", group_size);
    let group_count = batch_size / group_size + (batch_size % group_size != 0) as i64;
    println!("group_count: {}", group_count);
    let batch_idx = Tensor::arange(batch_size as i64, (Kind::Int64, image.device())).remainder(group_size);
    let batch_idx = batch_idx.view([-1, 1, 1, 1]);
    let pairs = batch_idx * num_shades.pow(2) + ref_img * num_shades as i64 + neigh_img;

    let glcms = {
        // We split the tensor into groups of size group_size.
        let pairs = pairs.tensor_split(group_count, 0);
        let bincount_size = num_shades.pow(2) * group_size;
        pairs.iter()
            .map(|t| t.view([-1]))
            .map(|t| t.bincount::<&Tensor>(None, bincount_size))
            .map(|t| t.view([-1, num_shades as i64, num_shades as i64]))
            .collect::<Vec<_>>()
    };
    let mut glcm = Tensor::cat(&glcms[..], 0).i((.., ..num_shades-1, ..num_shades-1));

    if symmetric {
        glcm+=glcm.copy().transpose(-1, -2);
    }

    let len = glcm.sum_dim_intlist(Some(&[1, 2][..]), false, Kind::Float);
    let len = len.view([-1, 1, 1]);
    &glcm / len
}

pub fn glcm_cpu(image: &Tensor, offset: (i64, i64), num_shades: u8, mask: Option<&Tensor>, symmetric: bool) -> Tensor{
    let (offset_y, offset_x) = offset;

    let (batch_size, _, height, width) = image.size4().unwrap();
    let (batch_size, height, width) = (batch_size as usize, height as usize, width as usize);
    let device = image.device();

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
    use tch::{Tensor, Kind, index::*};

    use crate::utils::assert_eq_tensor;

    #[test]
    fn bidouillage(){
        
    }

    #[test]
    fn test_glcm_no_mask(){
        let input = Tensor::of_slice(&[
            1.0, 1.0, 2.0, 1.0,
            2.0, 1.0, 1.0, 1.0,
            3.0, 2.0, 3.0, 1.0,
            3.0, 2.0, 1.0, 2.0,             

            1.0, 1.0, 2.0, 1.0,
            2.0, 1.0, 1.0, 1.0,
            3.0, 2.0, 3.0, 1.0,
            3.0, 2.0, 1.0, 2.0
        ]);

        let expected = Tensor::of_slice(&[
            0.25, 0.17, 0.0,
            0.25, 0.0,  0.08,
            0.08, 0.17, 0.0,

            0.25, 0.17, 0.0,
            0.25, 0.0,  0.08,
            0.08, 0.17, 0.0,
        ]).view((2, 3, 3));
        
        let input = (input.view((2, 1, 4, 4))-1.0) / 3.0;
        let rand = Tensor::rand(&[2, 1, 4, 4], (Kind::Float, tch::Device::Cpu));
        let input = Tensor::cat(&[input, rand], 0);
        let glcm = super::glcm(&input, (1, 0), 3, None, false);
        let glcm_gpu = super::glcm_gpu(&input, (1, 0), 3, None, false);
        let glcm_cpu = super::glcm_cpu(&input, (1, 0), 3, None, false);

        let glcm = glcm.i(..2);
        let glcm_gpu = glcm_gpu.i(..2);
        let glcm_cpu = glcm_cpu.i(..2);


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