/*!
Utilities for the utility crate :D
 */

use std::{path::{ PathBuf}, str::FromStr};

use tch::Tensor;

#[cfg(feature = "ndarray")]
use crate::ndarray::NDATensorExt;

pub fn assert_eq_tensor(a: &Tensor, b: &Tensor) {
    assert_eq!(a.size(), b.size(), "Tensors must have the same shape");
    let delta = f64::from((a - b).sum(tch::Kind::Float));
    assert!(delta < 1e-5, "Tensors must be equal (delta: {})", delta);
}

pub fn assert_eq_tensor_d(a: &Tensor, b: &Tensor, max_delta: f64) {
    assert_eq!(a.size(), b.size(), "Tensors must have the same shape");
    let delta = f64::from((a - b).sum(tch::Kind::Float));
    assert!(delta < max_delta, "Tensors must be equal (delta: {})", delta);
}

pub fn dirty_load(path: &str) -> Tensor {
    let path = PathBuf::from_str(path).unwrap();
    if !path.exists(){
        panic!("Asset not found: {:?}", path)
    }
    
    return match path.extension(){
        Some(ext) if ext == "pt" => Tensor::load(path).expect("Failed to load asset"),
        Some(ext) if ext == "png" || ext == "jpeg" || ext == "jpg" => tch::vision::image::load(path).expect("Failed to load asset"),
        Some(ext) if ext == "npy" => {
            #[cfg(feature = "ndarray")]
            {
                let array = ndarray_npy::read_npy(path).expect("Failed to load asset");
                Tensor::from_ndarray(array)
            }
            #[cfg(not(feature = "ndarray"))]
            {
                panic!("loading npy files requires ndarray feature")
            }
        }
        _ => panic!("Asset file unsupported: {:?}", path)
    };
}

pub fn assert_tensor_asset(tensor: &Tensor, asset: &str) {
    let asset = dirty_load(asset);
    assert_eq_tensor(tensor, &asset);
}