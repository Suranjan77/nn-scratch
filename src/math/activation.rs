//! Activation functions and their derivatives.
//!
//! An activation function is applied to every value of a layer's pre-activation
//! `A = X·W + b`, each value on its own (softmax is the exception: it works
//! row by row, see below).
//!
//! Without one, two layers are no better than one: two matrix multiplications in
//! a row are the same as one multiplication by their product,
//! `(X·W1)·W2 = X·(W1·W2)`. So something that is NOT a multiplication must sit
//! between the layers, letting the next layer depend on which features fire,
//! not just on how much ink there is.
//!
//! The derivatives (`d_*`) are used during back-propagation to find out how much
//! a small change in the pre-activation changes the output of the activation.

use crate::math::matrix::Matrix;

/// ReLU: `max(0, x)`. Keeps the positives and zeroes the rest:
/// a feature is either on (and says how strongly) or off.
pub fn relu(m: &Matrix) -> Matrix {
    let mut data = vec![0_f64; m.data().len()];

    for i in 0..m.data().len() {
        data[i] = 0_f64.max(m.data()[i]);
    }

    Matrix::new(m.rows, m.cols, data)
}

/// Derivative of ReLU: 1 where `x > 0`, otherwise 0.
/// A feature that was off passed nothing forward, so it takes no blame back.
pub fn d_relu(m: &Matrix) -> Matrix {
    let mut data = vec![0_f64; m.data().len()];

    for i in 0..m.data().len() {
        data[i] = match m.data()[i] {
            x if x > 0.0 => 1.0,
            _ => 0.0,
        }
    }

    Matrix::new(m.rows, m.cols, data)
}

/// Sigmoid: `1 / (1 + e^-x)`. Squashes any number into the range (0, 1):
/// a smooth version of "off" (near 0) and "on" (near 1).
pub fn sigmoid(m: &Matrix) -> Matrix {
    let mut data = vec![0_f64; m.data().len()];

    for i in 0..m.data().len() {
        data[i] = 1.0 / (1.0 + std::f64::consts::E.powf(-m.data()[i]));
    }

    Matrix::new(m.rows, m.cols, data)
}

/// Derivative of sigmoid: `sigmoid(x) * (1 - sigmoid(x))`.
/// It is largest (0.25) at `x = 0` and close to 0 for very large or very small
/// `x`, which is why inputs must be scaled and starting weights kept small.
pub fn d_sigmoid(m: &Matrix) -> Matrix {
    let one = Matrix::repeat(m.rows, m.cols, 1.0);
    let s = sigmoid(m);
    let r = &one - &s;
    &s * &r
}

/// Softmax, applied to every ROW on its own (one row = one image's ten scores):
/// `e^s_k / (e^s_0 + ... + e^s_9)`. Ten scores become ten probabilities:
/// `exp` makes every score positive, and dividing by the row total makes them add up to 1.
/// The highest score gets the highest probability, so the `argmax` of a row is the
/// predicted digit.
///
/// Example: scores where digit 7 has 7.54 and the next best, 9, has 2.33.
/// `e^7.54 = 1875.6` out of a total of about 1892.7, so `p(7) = 0.991`.
///
/// Softmax is almost always paired with the cross-entropy loss. Together their
/// combined derivative is simply `prediction - label`, so we never need a separate
/// `d_softmax` (the output layer is created with `None` as its derivative).
///
/// The row's maximum is subtracted before `exp()` for numerical stability:
/// `e^1000` overflows to infinity, but `e^(1000 - 1000) = 1` is fine. Subtracting the
/// same number from every score of a row does not change its probabilities.
pub fn softmax(m: &Matrix) -> Matrix {
    let mut data = Vec::with_capacity(m.rows * m.cols);

    for i in 0..m.rows {
        let row: Vec<f64> = (0..m.cols).map(|j| m.get(i, j)).collect();
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|s| (s - max).exp()).collect();
        let total: f64 = exps.iter().sum();
        data.extend(exps.iter().map(|e| e / total));
    }

    Matrix::new(m.rows, m.cols, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let m = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
        let r = relu(&m);
        assert!(r.data().iter().all(|&x| x >= 0.0));

        let m = Matrix::new(2, 2, vec![1.0, -1.0, -1.0, -231.0]);
        let r = relu(&m);
        assert!(r.data().iter().all(|&x| x >= 0.0));

        let m = Matrix::new(2, 2, vec![0.0, -1.0, -1.0, -231.0]);
        let r = relu(&m);
        assert!(r.data().iter().all(|&x| x >= 0.0));
    }

    /// Helper function to compare two vectors of floats for approximate equality.
    /// Standard `assert_eq!` will fail due to floating-point inaccuracies.
    fn assert_vec_approx_eq(a: &[f64], b: &[f64]) {
        let epsilon = 1e-9; // A small tolerance for floating point comparison

        assert_eq!(a.len(), b.len(), "Test vectors have different lengths.");

        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            if (val_a - val_b).abs() > epsilon {
                panic!(
                    "Vectors differ at index {}: left = {}, right = {}",
                    i, val_a, val_b
                );
            }
        }
    }

    #[test]
    fn test_sigmoid_zero() {
        // The sigmoid of 0.0 should be exactly 0.5
        let m = Matrix::new(1, 1, vec![0.0]);
        let s = sigmoid(&m);
        let expected = vec![0.5];
        assert_vec_approx_eq(&s.data(), &expected);
    }

    #[test]
    fn test_sigmoid_positive_values() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let s = sigmoid(&m);
        let expected = vec![
            0.7310585786300049, // f(1.0)
            0.8807970779778823, // f(2.0)
            0.9525741268224334, // f(3.0)
            0.9820137900379085, // f(4.0)
        ];
        assert_vec_approx_eq(&s.data(), &expected);
    }

    #[test]
    fn test_sigmoid_negative_values() {
        let mut m = Matrix::new(1, 3, vec![-1.0, -2.0, -3.0]);
        let s = sigmoid(&mut m);
        let expected = vec![
            0.2689414213699951,  // f(-1.0)
            0.11920292202211755, // f(-2.0)
            0.04742587317756678, // f(-3.0)
        ];
        assert_vec_approx_eq(&s.data(), &expected);
    }

    #[test]
    fn test_sigmoid_mixed_values() {
        let m = Matrix::new(3, 1, vec![1.5, 0.0, -2.5]);
        let s = sigmoid(&m);
        let expected = vec![
            0.8175744761936437,  // f(1.5)
            0.5,                 // f(0.0)
            0.07585818002124355, // f(-2.5)
        ];
        assert_vec_approx_eq(&s.data(), &expected);
    }

    #[test]
    fn test_sigmoid_saturation_large_values() {
        let mut m = Matrix::new(1, 2, vec![20.0, -20.0]);
        let s = sigmoid(&mut m);

        let expected_pos = 0.9999999979388463;
        let expected_neg = 2.0611536224385576e-9;

        let expected = vec![expected_pos, expected_neg];
        assert_vec_approx_eq(&s.data(), &expected);

        assert!(s.data()[0] > 0.99999999);
        assert!(s.data()[1] < 1e-8);
    }

    #[test]
    fn test_sigmoid_empty_matrix() {
        let m = Matrix::new(0, 0, vec![]);
        let s = sigmoid(&m);
        let expected: Vec<f64> = vec![];

        assert_eq!(s.data(), expected);
    }
}
