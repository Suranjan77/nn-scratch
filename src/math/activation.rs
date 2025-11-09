use crate::math::matrix::Matrix;

#[allow(dead_code)]
pub fn relu(m: &Matrix) -> Matrix {
    let mut data = vec![0_f64; m.data.len()];

    for i in 0..m.data.len() {
        data[i] = 0_f64.max(m.data[i]);
    }

    Matrix::new(m.rows(), m.cols(), data)
}

#[allow(dead_code)]
pub fn d_relu(m: &Matrix) -> Matrix {
    let mut data = vec![0_f64; m.data.len()];

    for i in 0..m.data.len() {
        data[i] = match m.data[i] {
            x if x > 0.0 => 1.0,
            _ => 0.0,
        }
    }

    Matrix::new(m.rows(), m.cols(), data)
}

#[allow(dead_code)]
pub fn sigmoid(m: &Matrix) -> Matrix {
    let mut data = vec![0_f64; m.data.len()];

    for i in 0..m.data.len() {
        data[i] = 1.0 / (1.0 + std::f64::consts::E.powf(-m.data[i]));
    }

    Matrix::new(m.rows(), m.cols(), data)
}

#[allow(dead_code)]
pub fn d_sigmoid(m: &Matrix) -> Matrix {
    let one = Matrix::repeat(m.rows(), m.cols(), 1.0);
    let s = sigmoid(m);
    let r = &one - &s;
    &s * &r
}

// Softmax is almost always paired with cross-entropy loss function to prevent vanishing gradient and also allows calculation of gradient in simpler way.
// y_hat - y is the gradient so, no need to explicitly calculate derivative of softmax.
#[allow(dead_code)]
pub fn softmax(m: &Matrix) -> Matrix {
    let mut data = vec![0_f64; m.data.len()];
    let sum = m
        .data
        .iter()
        .map(|x| x.exp())
        .reduce(|acc, x| acc + x)
        .unwrap_or(0.0);
    for i in 0..m.data.len() {
        data[i] = m.data[i] / sum;
    }

    Matrix::new(m.rows(), m.cols(), data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let m = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
        let r = relu(&m);
        assert!(r.data.iter().all(|&x| x >= 0.0));

        let m = Matrix::new(2, 2, vec![1.0, -1.0, -1.0, -231.0]);
        let r = relu(&m);
        assert!(r.data.iter().all(|&x| x >= 0.0));

        let m = Matrix::new(2, 2, vec![0.0, -1.0, -1.0, -231.0]);
        let r = relu(&m);
        assert!(r.data.iter().all(|&x| x >= 0.0));
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
        assert_vec_approx_eq(&s.data, &expected);
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
        assert_vec_approx_eq(&s.data, &expected);
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
        assert_vec_approx_eq(&s.data, &expected);
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
        assert_vec_approx_eq(&s.data, &expected);
    }

    #[test]
    fn test_sigmoid_saturation_large_values() {
        let mut m = Matrix::new(1, 2, vec![20.0, -20.0]);
        let s = sigmoid(&mut m);

        let expected_pos = 0.9999999979388463;
        let expected_neg = 2.0611536224385576e-9;

        let expected = vec![expected_pos, expected_neg];
        assert_vec_approx_eq(&s.data, &expected);

        assert!(s.data[0] > 0.99999999);
        assert!(s.data[1] < 1e-8);
    }

    #[test]
    fn test_sigmoid_empty_matrix() {
        let m = Matrix::new(0, 0, vec![]);
        let s = sigmoid(&m);
        let expected: Vec<f64> = vec![];

        assert_eq!(s.data, expected);
    }
}
