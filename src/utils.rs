use tch::Tensor;

pub fn assert_eq_tensor(a: &Tensor, b: &Tensor) {
    assert_eq!(a.size(), b.size(), "Tensors must have the same shape");
    let delta = f64::from((a - b).sum(tch::Kind::Float));
    assert!(delta < 1e-5, "Tensors must be equal");
}

pub fn assert_tensor_asset(tensor: &Tensor, asset: &str) {
    let asset =Tensor::load(asset).expect("Failed to load asset");
    assert_eq_tensor(tensor, &asset);
}