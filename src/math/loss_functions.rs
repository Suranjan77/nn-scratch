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

// Suitable for multi-class classification
pub fn cross_entropy(prediction: &Matrix, actual: &Matrix) -> f64 {
    if actual.rows() != prediction.rows() {
        panic!("Dimensions do not match");
    }

    let mut loss = 0.0;
    for i in 0..actual.rows() {
        let p = prediction.data[i];
        let y = actual.data[i];
        loss += y * p.ln();
    }

    -loss
}

// Suitable for binary classification
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
