use tch::Tensor;

const RGB_FROM_HED : &[f32; 9] = &[
    0.65, 0.70, 0.29,
    0.07, 0.99, 0.11,
    0.27, 0.57, 0.78
];

/**
Convert a tensor of HED values to RGB values.

# Arguments
- hed: Tensor - The tensor of HED values [N, 3, H, W] with float values in the range [0, 1]

# Returns
Tensor - The tensor of RGB values [N, 3, H, W] with float values in the range [0, 1]
 */
pub fn rgb_from_hed(hed: &Tensor) -> Tensor {
    let rgb_from_hed = Tensor::of_slice(RGB_FROM_HED).view([3, 3]);
    hed.transpose(1, 3).matmul(&rgb_from_hed).transpose(1, 3)
}

/**
Convert a tensor of RGB values to HED values.

# Arguments
- rgb: Tensor - The tensor of RGB values [N, 3, H, W] with float values in the range [0, 1]

# Returns
Tensor - The tensor of HED values [N, 3, H, W] with float values in the range [0, 1]
 */
pub fn hed_from_rgb(rgb: &Tensor) -> Tensor {
    let rgb_from_hed = Tensor::of_slice(RGB_FROM_HED).view([3, 3]);
    let hed_from_rgb = rgb_from_hed.inverse();
    rgb.transpose(1, 3).matmul(&hed_from_rgb).transpose(1, 3)
}

/**
Convert a tensor of RGB values to HSV values.

# Arguments
- rgb: Tensor - The tensor of RGB values [N, 3, H, W] with float values in the range [0, 1]

# Returns
Tensor - The tensor of HSV values [N, 3, H, W] with float values in the range [0, 1]
 */
pub fn hsv_from_rgb(rgb: &Tensor) -> Tensor {
    let (max /* [N, H, W] */, _) = rgb.max_dim(1, false);
    let (min /* [N, H, W] */, _) = rgb.min_dim(1, false); 
    let delta /* [N, H, W] */ = &max - min; 
    let mut h  /* [N, H, W] */ = Tensor::zeros_like(&delta); 
    h += rgb.select(1, 0).eq_tensor(&delta) 
        * ((rgb.select(1, 1) - rgb.select(1, 2)) / &delta).fmod(6.0) * 60.0;
    h += rgb.select(1, 1).eq_tensor(&delta) 
        * ((rgb.select(1, 2) - rgb.select(1, 0)) / &delta + 2.0) * 60.0;
    h += rgb.select(1, 2).eq_tensor(&delta) 
        * ((rgb.select(1, 0) - rgb.select(1, 1)) / &delta + 4.0) * 60.0;
    h = h.fmod(360.0);
    let s  /* [N, H, W] */ = &delta / &max; 
    let s = s.where_scalarother(&max.eq_tensor(&Tensor::zeros_like(&max)), 0);
    let v  /* [N, H, W] */ = max; 

    Tensor::stack(&[h, s, v], 1)
}
