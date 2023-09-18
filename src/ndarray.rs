use tch::Kind;

pub trait NDATensorExt {
    fn to_ndarray(&self) -> ndarray::ArrayD<f32>;

    fn from_ndarray(array: ndarray::ArrayD<f32>) -> Self;
}

impl NDATensorExt for tch::Tensor {
    fn to_ndarray(&self) -> ndarray::ArrayD<f32> {
        let dims = self.size();
        let casted = self.to_kind(Kind::Float);
        let data = Vec::<f32>::from(&casted);
        match dims.len() {
            0 => ndarray::arr0(data[0]).into_dyn(),
            1 => ndarray::Array1::from(data).into_dyn(),
            2 => ndarray::Array2::from_shape_vec((dims[0] as usize, dims[1] as usize), data)
                .unwrap()
                .into_dyn(),
            3 => ndarray::Array3::from_shape_vec(
                (dims[0] as usize, dims[1] as usize, dims[2] as usize),
                data,
            )
            .unwrap()
            .into_dyn(),
            4 => ndarray::Array4::from_shape_vec(
                (
                    dims[0] as usize,
                    dims[1] as usize,
                    dims[2] as usize,
                    dims[3] as usize,
                ),
                data,
            )
            .unwrap()
            .into_dyn(),
            5 => ndarray::Array5::from_shape_vec(
                (
                    dims[0] as usize,
                    dims[1] as usize,
                    dims[2] as usize,
                    dims[3] as usize,
                    dims[4] as usize,
                ),
                data,
            )
            .unwrap()
            .into_dyn(),
            6 => ndarray::Array6::from_shape_vec(
                (
                    dims[0] as usize,
                    dims[1] as usize,
                    dims[2] as usize,
                    dims[3] as usize,
                    dims[4] as usize,
                    dims[5] as usize,
                ),
                data,
            )
            .unwrap()
            .into_dyn(),
            _ => panic!("Unsupported tensor shape"),
        }
    }

    fn from_ndarray(array: ndarray::ArrayD<f32>) -> Self {
        let shape = array.shape().to_owned();
        let mut l = 1;
        for i in shape.iter() {
            l *= i;
        }
        let tensor = tch::Tensor::of_slice(&array.into_shape((l,)).unwrap().to_vec());
        tensor.reshape(
            shape
                .iter()
                .map(|x| *x as i64)
                .collect::<Vec<i64>>()
                .as_slice(),
        )
    }
}
