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

pub fn binary_cross_entropy(prediction: &Matrix, actual: &Matrix) -> f64 {
    if actual.rows() != prediction.rows() {
        panic!("Dimensions do not match");
    }

    let mut loss = 0.0;
    for i in 0..actual.rows() {
        let p = prediction.data[i];
        let y = actual.data[i];
        loss -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
    }

    loss
}

pub fn d_binary_cross_entropy(prediction: &Matrix, actual: &Matrix) -> Matrix {
    let nominator = prediction - actual;
    let denominator =
        prediction * &(&Matrix::repeat(prediction.rows(), prediction.cols(), 1.0) - prediction);
    &nominator * &denominator.powi(-1)
}
