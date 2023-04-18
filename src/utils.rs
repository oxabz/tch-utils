/*!
Utilities for the utility crate :D
 */

use std::{path::{ PathBuf}, str::FromStr, ffi::OsStr};

use tch::Tensor;

#[cfg(feature = "ndarray")]
use crate::ndarray::TensorExt;

pub fn assert_eq_tensor(a: &Tensor, b: &Tensor) {
    assert_eq!(a.size(), b.size(), "Tensors must have the same shape");
    let delta = f64::from((a - b).sum(tch::Kind::Float));
    assert!(delta < 1e-5, "Tensors must be equal");
}

pub fn assert_tensor_asset(tensor: &Tensor, asset: &str) {
    let path = PathBuf::from_str(asset).unwrap();
    if !path.exists(){
        panic!("Asset not found: {}", asset)
    }
    
    let asset: Tensor = match path.extension(){
        Some(ext) if ext == "pt" => Tensor::load(asset).expect("Failed to load asset"),
        Some(ext) if ext == "png" || ext == "jpeg" || ext == "jpg" => tch::vision::image::load(asset).expect("Failed to load asset"),
        Some(ext) if ext == "npy" => {
            #[cfg(feature = "ndarray")]
            {
                let array = ndarray_npy::read_npy(asset).expect("Failed to load asset");
                Tensor::from_ndarray(array)
            }
            #[cfg(not(feature = "ndarray"))]
            {
                panic!("loading npy files requires ndarray feature")
            }
        }
        _ => panic!("Asset file unsupported: {}", asset)
    };


    assert_eq_tensor(tensor, &asset);
}