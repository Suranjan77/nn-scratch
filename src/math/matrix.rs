use rand::distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Sub};

#[derive(Serialize, Deserialize, Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    pub data: Vec<f64>,
    pub transposed: bool,
}

impl Matrix {
    pub fn rows(&self) -> usize {
        if self.transposed {
            self.cols
        } else {
            self.rows
        }
    }

    pub fn cols(&self) -> usize {
        if self.transposed {
            self.rows
        } else {
            self.cols
        }
    }

    #[allow(dead_code)]
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        if rows * cols != data.len() {
            panic!("Data length does not match dimensions");
        }
        Self {
            data,
            rows,
            cols,
            transposed: false,
        }
    }

    #[allow(dead_code)]
    pub fn uniform(rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();
        let distribution = Uniform::try_from(-1.0..1.0).unwrap();
        let data = (0..rows * cols)
            .map(|_| distribution.sample(&mut rng))
            .collect();
        Matrix::new(rows, cols, data)
    }

    #[allow(dead_code)]
    pub fn eye(size: usize) -> Self {
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    data[i * size + j] = 1.0;
                } else {
                    data[i * size + j] = 0.0;
                }
            }
        }
        Matrix::new(size, size, data)
    }

    #[allow(dead_code)]
    pub fn repeat(rows: usize, cols: usize, repeat_value: f64) -> Self {
        let data = vec![repeat_value; rows * cols];
        Matrix::new(rows, cols, data)
    }

    #[allow(dead_code)]
    pub fn transpose(&mut self) {
        self.transposed = !self.transposed;
    }

    #[allow(dead_code)]
    pub fn dot(&self, other: &Matrix) -> Result<Matrix, &'static str> {
        if self.cols() != other.rows() {
            return Err("Matrix multiplication dimension mismatch: A.cols != B.rows");
        }

        let mut res = Matrix::repeat(self.rows(), other.cols(), 0.0);

        for i in 0..self.rows() {
            for j in 0..other.cols() {
                let mut sum = 0.0;
                for k in 0..self.cols() {
                    sum += self.data[i * self.cols() + k] * other.data[k * other.cols() + j];
                }
                res.data[i * other.cols() + j] = sum;
            }
        }

        Ok(res)
    }

    #[allow(dead_code)]
    pub fn powi(&self, exp: i32) -> Self {
        let mut data = vec![0.0; self.cols() * self.rows];
        for i in 0..data.len() {
            data[i] = self.data[i].powi(exp);
        }
        Matrix::new(self.rows(), self.cols(), data)
    }
}

impl Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Matrix {
        if other.data.len() == 1 {
            let mut data = vec![0.0; self.data.len()];
            for i in 0..self.data.len() {
                data[i] = self.data[i] * other.data[0];
            }
            return Matrix::new(self.rows(), self.cols(), data);
        }

        if self.cols() != other.cols() {
            panic!("Dimensions are not compatible: A.cols != B.cols");
        }

        if (self.rows() == other.rows()) || (other.rows() == 1) {
            let mut data = vec![0.0; self.data.len()];
            if other.rows() == self.rows() {
                for i in 0..self.rows() {
                    for j in 0..self.cols() {
                        data[i * self.cols() + j] =
                            self.data[i * self.cols() + j] * other.data[i * self.cols() + j];
                    }
                }
            } else {
                for i in 0..self.rows() {
                    for j in 0..self.cols() {
                        data[i * self.cols() + j] = self.data[i * self.cols() + j] * other.data[j];
                    }
                }
            }
            Matrix::new(self.rows(), self.cols(), data)
        } else {
            panic!("Dimension are not compatible: either A.rows != B.rows or B.rows != 1");
        }
    }
}

impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Matrix {
        if other.data.len() == 1 {
            let mut data = vec![0.0; self.data.len()];
            for i in 0..self.data.len() {
                data[i] = self.data[i] + other.data[0];
            }
            return Matrix::new(self.rows(), self.cols(), data);
        }

        if self.cols() != other.cols() {
            panic!("Dimensions are not compatible: A.cols != B.cols");
        }

        if (self.rows() == other.rows()) || (other.rows() == 1) {
            let mut data = vec![0.0; self.data.len()];
            if other.rows() == self.rows() {
                for i in 0..self.rows() {
                    for j in 0..self.cols() {
                        data[i * self.cols() + j] =
                            self.data[i * self.cols() + j] + other.data[i * self.cols() + j];
                    }
                }
            } else {
                for i in 0..self.rows() {
                    for j in 0..self.cols() {
                        data[i * self.cols() + j] = self.data[i * self.cols() + j] + other.data[j];
                    }
                }
            }
            Matrix::new(self.rows(), self.cols(), data)
        } else {
            panic!("Dimension are not compatible: either A.rows != B.rows or B.rows != 1");
        }
    }
}

impl Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, other: &Matrix) -> Matrix {
        if other.data.len() == 1 {
            let mut data = vec![0.0; self.data.len()];
            for i in 0..self.data.len() {
                data[i] = self.data[i] - other.data[0];
            }
            return Matrix::new(self.rows(), self.cols(), data);
        }

        if self.cols() != other.cols() {
            panic!("Dimensions are not compatible: A.cols != B.cols");
        }

        if (self.rows() == other.rows()) || (other.rows() == 1) {
            let mut data = vec![0.0; self.data.len()];
            if other.rows() == self.rows() {
                for i in 0..self.rows() {
                    for j in 0..self.cols() {
                        data[i * self.cols() + j] =
                            self.data[i * self.cols() + j] - other.data[i * self.cols() + j];
                    }
                }
            } else {
                for i in 0..self.rows() {
                    for j in 0..self.cols() {
                        data[i * self.cols() + j] = self.data[i * self.cols() + j] - other.data[j];
                    }
                }
            }
            Matrix::new(self.rows(), self.cols(), data)
        } else {
            panic!("Dimension are not compatible: either A.rows != B.rows or B.rows != 1");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_by_tow() {
        let a = Matrix::new(2, 2, vec![2.1, 2.3, 1.2, 1.4]);
        let b = Matrix::new(2, 2, vec![1.3, 1.2, 2.9, 2.7]);

        let res = &a.dot(&b).unwrap();

        assert_eq!(res.rows(), a.rows());
        assert_eq!(res.cols(), b.cols());
        assert_eq!(
            res.data,
            vec![9.399999999999999, 8.73, 5.619999999999999, 5.22]
        );
    }

    #[test]
    fn three_by_one() {
        let a = Matrix::repeat(3, 3, 2.0);
        let b = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);

        let res = &a.dot(&b).unwrap();

        assert_eq!(res.rows(), a.rows());
        assert_eq!(res.cols(), b.cols());
        assert_eq!(res.data, vec![12.0, 12.0, 12.0]);
    }

    #[test]
    fn three_by_three() {
        let a = Matrix::repeat(3, 3, 2.0);
        let b = Matrix::repeat(3, 3, 2.0);

        let res = &a.dot(&b).unwrap();

        assert_eq!(res.rows(), a.rows());
        assert_eq!(res.cols(), b.cols());
        assert_eq!(res.data, vec![12.0; 9]);
    }

    #[test]
    fn four_by_four() {
        let a = Matrix::repeat(4, 4, 2.0);
        let b = Matrix::repeat(4, 4, 2.0);

        let res = &a.dot(&b).unwrap();

        assert_eq!(res.rows(), a.rows());
        assert_eq!(res.cols(), b.cols());
        assert_eq!(res.data, vec![16.0; 16]); // 4 * (2.0 * 2.0) = 16.0
    }

    #[test]
    fn one_twentyone_by_one_twentyone() {
        let a = Matrix::repeat(121, 121, 2.0);
        let b = Matrix::repeat(121, 121, 2.0);

        let res = &a.dot(&b).unwrap();

        assert_eq!(res.rows(), a.rows());
        assert_eq!(res.cols(), b.cols());
        assert_eq!(res.data, vec![484.0; 14641]); // 121 * (2.0 * 2.0) = 484.0
    }

    #[test]
    fn mul_rectangular_matrices() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let res = &a.dot(&b).unwrap();

        let expected_data = vec![58.0, 64.0, 139.0, 154.0];

        assert_eq!(res.rows(), 2);
        assert_eq!(res.cols(), 2);
        assert_eq!(res.data, expected_data);
    }

    #[test]
    fn mul_identity_matrices() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let i_3x3 = Matrix::new(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let i_2x2 = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);

        // Test A * I = A
        let res_a = &a.dot(&i_3x3).unwrap();
        assert_eq!(res_a.rows(), a.rows());
        assert_eq!(res_a.cols(), a.cols());
        assert_eq!(res_a.data, a.data);

        // Test I * A = A
        let res_b = &i_2x2.dot(&a).unwrap();
        assert_eq!(res_b.rows(), a.rows());
        assert_eq!(res_b.cols(), a.cols());
        assert_eq!(res_b.data, a.data);
    }

    #[test]
    fn mul_with_zeros_and_negatives() {
        let a = Matrix::new(2, 2, vec![-1.0, 0.0, 2.0, 3.0]);
        let b = Matrix::new(2, 2, vec![5.0, -2.0, 0.0, 4.0]);

        let res = &a.dot(&b).unwrap();

        let expected_data = vec![-5.0, 2.0, 10.0, 8.0];

        assert_eq!(res.rows(), 2);
        assert_eq!(res.cols(), 2);
        assert_eq!(res.data, expected_data);
    }

    #[test]
    fn mul_row_vec_by_col_vec_dot_product() {
        let a = Matrix::new(1, 3, vec![1., 2., 3.]);
        let b = Matrix::new(3, 1, vec![4., 5., 6.]);

        let res = &a.dot(&b).unwrap();

        let expected_data = vec![32.0];

        assert_eq!(res.rows(), 1);
        assert_eq!(res.cols(), 1);
        assert_eq!(res.data, expected_data);
    }

    #[test]
    fn mul_col_vec_by_row_vec_outer_product() {
        let a = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);
        let b = Matrix::new(1, 3, vec![4.0, 5.0, 6.0]);

        let res = &a.dot(&b).unwrap();

        let expected_data = vec![4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0];

        assert_eq!(res.rows(), 3);
        assert_eq!(res.cols(), 3);
        assert_eq!(res.data, expected_data);
    }

    #[test]
    #[should_panic]
    fn mul_mismatched_dimensions_should_panic() {
        let a = Matrix::repeat(2, 3, 1.0);
        let b = Matrix::repeat(2, 2, 1.0);

        let _res = &a.dot(&b).unwrap();
    }

    #[test]
    fn transpose_rectangular_matrix() {
        let mut a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        a.transpose();

        assert_eq!(a.rows(), 3);
        assert_eq!(a.cols(), 2);
        assert_eq!(a.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn transpose_col_vector_to_row_vector() {
        // A (3x1)
        let mut a = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);

        a.transpose();

        // A^T (1x3)
        let expected_data = vec![1.0, 2.0, 3.0];

        assert_eq!(a.rows(), 1);
        assert_eq!(a.cols(), 3);
        assert_eq!(a.data, expected_data);
    }

    #[test]
    fn transpose_row_vector_to_col_vector() {
        // A (1x3)
        let mut a = Matrix::new(1, 3, vec![1., 2., 3.]);

        a.transpose();

        // A^T (3x1)
        let expected_data = vec![1.0, 2.0, 3.0];

        assert_eq!(a.rows(), 3);
        assert_eq!(a.cols(), 1);
        assert_eq!(a.data, expected_data);
    }

    #[test]
    fn test_scalar_positive() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(1, 1, vec![2.0]);
        let res = &m * &b;
        assert_eq!(res.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_scalar_zero() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(1, 1, vec![0.0]);
        let res = &m * &b;
        assert_eq!(res.data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_scalar_one() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(1, 1, vec![1.0]);
        let res = &m * &b;
        assert_eq!(res.data, vec![1.0, 2.0, 3.0, 4.0]); // No change
    }

    #[test]
    fn test_scalar_negative() {
        let m = Matrix::new(1, 3, vec![10.0, -20.0, 30.0]);
        let b = Matrix::new(1, 1, vec![-0.5]);
        let res = &m * &b;
        assert_eq!(res.data, vec![-5.0, 10.0, -15.0]);
    }

    #[test]
    fn test_scalar_empty_matrix() {
        let m = Matrix::new(0, 3, vec![]); // 0x3 matrix
        let b = Matrix::new(1, 1, vec![10.0]);
        let res = &m * &b;
        let expected: Vec<f64> = vec![];
        assert_eq!(res.data, expected);
    }

    #[test]
    fn test_broadcast_as_scalar() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(1, 1, vec![3.0]);
        let res = &m * &b;
        assert_eq!(res.data, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    #[should_panic]
    fn test_broadcast_err_col_mismatch() {
        let m = Matrix::new(2, 3, vec![1.0; 6]);
        let b = Matrix::new(1, 2, vec![1.0; 2]); // 1x2
        let _ = &m * &b;
    }

    #[test]
    fn test_broadcast_row_broadcast() {
        let m = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(1, 2, vec![2.0, 3.0]);

        let res = &m * &b;

        assert_eq!(res.data, [2.0, 6.0, 6.0, 12.0, 10.0, 18.0]);
    }

    #[test]
    fn test_broadcast_element_wise() {
        let m = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![1.0; 6]);

        let res = &m * &b;

        assert_eq!(res.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_broadcast_double_multiply() {
        let m = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]);
        let b = Matrix::new(1, 3, vec![2.0, 3.0, 4.0]);

        let res = &m * &b;

        assert_eq!(res.data, [2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_broadcast_as_scalar_add() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(1, 1, vec![3.0]);
        let res = &m + &b;

        assert_eq!(res.data, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    #[should_panic]
    fn test_broadcast_err_col_mismatch_add() {
        let m = Matrix::new(2, 3, vec![1.0; 6]);
        let b = Matrix::new(1, 2, vec![1.0; 2]); // 1x2
        let _ = &m + &b;
    }

    #[test]
    fn test_broadcast_row_broadcast_add() {
        let m = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(1, 2, vec![2.0, 3.0]);

        let res = &m + &b;

        assert_eq!(res.data, [3.0, 5.0, 5.0, 7.0, 7.0, 9.0]);
    }

    #[test]
    fn test_broadcast_element_wise_add() {
        let m = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![1.0; 6]);

        let res = &m + &b;

        assert_eq!(res.data, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_broadcast_double_add() {
        let m = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]);
        let b = Matrix::new(1, 3, vec![2.0, 3.0, 4.0]);

        let res = &m + &b;

        assert_eq!(res.data, [3.0, 5.0, 7.0]);
    }
}
