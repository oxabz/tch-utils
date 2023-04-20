use tch::Tensor;

pub fn glcm(
    image: &Tensor,
    offset: (i64, i64),
    num_colors: i64,
    mask: Option<&Tensor>,
) -> Tensor{
    todo!()
}