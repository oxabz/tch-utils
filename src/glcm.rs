use tch::Tensor;

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
 */
pub fn glcm(
    image: &Tensor,
    offset: (i64, i64),
    num_shades: u8,
    mask: Option<&Tensor>,
) -> Tensor{
    todo!()
}