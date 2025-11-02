use crate::math::matrix::Matrix;

pub fn sse(prediction: &Matrix, actual: &Matrix) -> f64 {
    if actual.rows() != prediction.rows() {
        panic!("Dimensions do not match");
    }

    let mut loss = 0.0;
    for i in 0..actual.rows() {
        loss += (prediction.data[i] - actual.data[i]).powi(2);
    }

    loss
}
