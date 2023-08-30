/*!
Utilities for the utility crate :D
 */

use std::{path::PathBuf, str::FromStr};

use tch::Tensor;

#[cfg(feature = "ndarray")]
use crate::ndarray::NDATensorExt;


pub fn graham_scan(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut points = points.to_vec();

    let mut min_y = points[0].1;
    let mut min_x = points[0].0;
    let mut min_i = 0;
    for (i, point) in points.iter().enumerate() {
        if point.1 < min_y {
            min_y = point.1;
            min_x = point.0;
            min_i = i;
        } else if point.1 == min_y && point.0 < min_x {
            min_x = point.0;
            min_i = i;
        }
    }
    let p = points.remove(min_i);
    
    let mut points = points
        .into_iter()
        .map(|(x, y)|((x, y), x - p.0, y - p.1))
        .map(|(p, dx, dy)| (p, (dx.powi(2) + dy.powi(2)).sqrt(), dx))
        .map(|(p, len, dot)| (p, len, - dot / len))
        .filter(|(_, len, _)| *len > 0.0)
        .collect::<Vec<_>>();

    points.sort_by(|(_, al, aa), (_, bl, ba)| aa.partial_cmp(ba).unwrap().then(al.partial_cmp(bl).unwrap()));

    let mut hull = vec![p];
    for (p, _, _) in points {
        while hull.len() > 1 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            let c = p;
            let ab = (b.0 - a.0, b.1 - a.1);
            let bc = (c.0 - b.0, c.1 - b.1);
            let cross = ab.0 * bc.1 - ab.1 * bc.0;
            if cross > 0.0 {
                break;
            }
            hull.pop();
        }
        hull.push(p);
    }

    hull 
    
}

pub fn assert_eq_tensor(a: &Tensor, b: &Tensor) {
    assert_eq_tensor_d(a, b, 1e-5);
}

pub fn assert_eq_tensor_d(a: &Tensor, b: &Tensor, max_delta: f64) {
    assert_eq!(a.size(), b.size(), "Tensors must have the same shape");
    let delta = f64::from((a - b).abs().sum(tch::Kind::Float));
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

pub fn assert_tensor_asset_d(tensor: &Tensor, asset: &str, max_delta: f64) {
    let asset = dirty_load(asset);
    assert_eq_tensor_d(tensor, &asset, max_delta);
}