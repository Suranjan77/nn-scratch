//! Loss functions: one number for how wrong the predictions are.
//! A lower loss means better predictions.
//!
//! All of them take two matrices of the same shape, one row per image:
//! `prediction` (what the network said) and `actual` (the right answers),
//! and return the loss AVERAGED over the images (rows).

use crate::math::matrix::Matrix;

/// Smallest probability we allow inside `ln()`.
/// `ln(0)` is `-inf`, which would make the whole loss `inf`/`NaN`.
const EPSILON: f64 = 1e-12;

fn check_shapes(prediction: &Matrix, actual: &Matrix) {
    if prediction.rows != actual.rows || prediction.cols != actual.cols {
        panic!("Dimensions do not match");
    }
}

/// Sum of squared errors per image, averaged: `mean over rows of sum((p - y)^2)`.
/// Suitable for regression problems.
pub fn sse(prediction: &Matrix, actual: &Matrix) -> f64 {
    check_shapes(prediction, actual);

    let mut loss = 0.0;
    for i in 0..actual.rows {
        for j in 0..actual.cols {
            loss += (prediction.get(i, j) - actual.get(i, j)).powi(2);
        }
    }

    loss / actual.rows as f64
}

/// Categorical cross-entropy: `mean over rows of -sum(y * ln(p))`.
///
/// Suitable for multi-class classification together with a softmax output.
/// With one-hot answers every term but one is 0, so for each image this is simply
/// `-ln(probability given to the true digit)`: one number for how wrong.
///
/// - an untrained network guesses about 0.1 for every digit: `-ln(0.1) = 2.30`
/// - a 50/50 guess on the true digit:                         `-ln(0.5) = 0.69`
/// - a confident right answer, 0.99:                          `-ln(0.99) = 0.01`
///
/// A confident WRONG answer costs the most: as the probability of the true digit
/// falls towards 0, the loss rises without limit.
pub fn cross_entropy(prediction: &Matrix, actual: &Matrix) -> f64 {
    check_shapes(prediction, actual);

    let mut loss = 0.0;
    for i in 0..actual.rows {
        for j in 0..actual.cols {
            let p = prediction.get(i, j).max(EPSILON);
            loss -= actual.get(i, j) * p.ln();
        }
    }

    loss / actual.rows as f64
}

/// Binary cross-entropy: `mean over rows of -sum(y * ln(p) + (1 - y) * ln(1 - p))`.
/// Suitable for binary (yes/no) classification with a sigmoid output.
pub fn binary_cross_entropy(prediction: &Matrix, actual: &Matrix) -> f64 {
    check_shapes(prediction, actual);

    let mut loss = 0.0;
    for i in 0..actual.rows {
        for j in 0..actual.cols {
            let p = prediction.get(i, j).clamp(EPSILON, 1.0 - EPSILON);
            let y = actual.get(i, j);
            loss -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
        }
    }

    loss / actual.rows as f64
}
