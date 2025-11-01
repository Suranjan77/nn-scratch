use crate::math::matrix::Matrix;

pub fn mse(prediction: &Matrix, actual: &Matrix) -> Matrix {
    if actual.rows != prediction.rows {
        panic!("Dimensions do not match");
    }

    let mut data = vec![0.0; prediction.rows];
    for i in 0..actual.rows {
        data[i] = (prediction.data[i] - actual.data[i]).powi(2) / 2.0;
    }

    Matrix::new(prediction.rows, 1, data)
}
