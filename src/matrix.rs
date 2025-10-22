
#[derive(Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

use std::ops::Mul;

impl<'a> Mul for &'a Matrix {
    type Output = Result<Matrix, String>;

    fn mul(self, other: &'a Matrix) -> Result<Matrix, String> {

        if self.cols != other.rows {
            return Err("Matrix multiplication dimension mismatch: A.cols != B.rows".to_string());
        }

        let mut res = Matrix {
            rows: self.rows,
            cols: other.cols,
            data: vec![0.0; self.rows * other.cols],
        };

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                res.data[i * other.cols + j] = sum;
            }
        }

        Ok(res)
    }
}

impl Matrix {
    #[allow(dead_code)]
    fn transpose(&mut self) {
        let mut transposed = vec![0.0; self.rows * self.cols];
        let mut c = 0;
        for i in 0..self.cols {
            for j in 0..self.rows {
                transposed[c] = self.data[j * self.cols + i];
                c += 1;
            }
        }
        let temp_rows = self.rows;
        self.rows = self.cols;
        self.cols = temp_rows;
        self.data = transposed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_by_tow() {
        let a = Matrix {rows: 2, cols: 2, data: vec![2.1, 2.3, 1.2, 1.4]};
        let b = Matrix {rows: 2, cols: 2, data: vec![1.3, 1.2, 2.9, 2.7]};

        let res = (&a * &b).unwrap(); 

        assert_eq!(res.rows, a.rows);
        assert_eq!(res.cols, b.cols);
        assert_eq!(res.data, vec![9.399999999999999, 8.73, 5.619999999999999, 5.22]);
    }

    #[test]
    fn three_by_one() {
        let a = Matrix {rows: 3, cols: 3, data: vec![2.0;9]};
        let b = Matrix {rows: 3, cols: 1, data: vec![1.0, 2.0, 3.0]};

        let res = (&a * &b).unwrap();

        assert_eq!(res.rows, a.rows);
        assert_eq!(res.cols, b.cols);
        assert_eq!(res.data, vec![12.0, 12.0, 12.0]);
    }

    #[test]
    fn three_by_three() {
        let a = Matrix {rows: 3, cols: 3, data: vec![2.0;9]};
        let b = Matrix {rows: 3, cols: 3, data: vec![2.0;9]};

        let res = (&a * &b).unwrap();

        assert_eq!(res.rows, a.rows);
        assert_eq!(res.cols, b.cols);
        assert_eq!(res.data, vec![12.0;9]);
    }

    #[test]
    fn transpose() {
        let mut a = Matrix {rows: 3, cols: 3, data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]};
        a.transpose();
        assert_eq!(a.data, vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn mul_rectangular_matrices() {
        let a = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 
                       4.0, 5.0, 6.0],
        };
        let b = Matrix {
            rows: 3,
            cols: 2,
            data: vec![7.0, 8.0, 
                       9.0, 10.0, 
                       11.0, 12.0],
        };

        let res = (&a * &b).unwrap();

        let expected_data = vec![58.0, 64.0, 139.0, 154.0];

        assert_eq!(res.rows, 2);
        assert_eq!(res.cols, 2);
        assert_eq!(res.data, expected_data);
    }

    #[test]
    fn mul_identity_matrices() {
        let a = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 
                       4.0, 5.0, 6.0],
        };
        let i_3x3 = Matrix {
            rows: 3,
            cols: 3,
            data: vec![1.0, 0.0, 0.0, 
                       0.0, 1.0, 0.0, 
                       0.0, 0.0, 1.0],
        };
        let i_2x2 = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 0.0, 
                       0.0, 1.0],
        };

        // Test A * I = A
        let res_a = (&a * &i_3x3).unwrap();
        assert_eq!(res_a.rows, a.rows);
        assert_eq!(res_a.cols, a.cols);
        assert_eq!(res_a.data, a.data);

        // Test I * A = A
        let res_b = (&i_2x2 * &a).unwrap();
        assert_eq!(res_b.rows, a.rows);
        assert_eq!(res_b.cols, a.cols);
        assert_eq!(res_b.data, a.data);
    }

    #[test]
    fn mul_with_zeros_and_negatives() {
        let a = Matrix {
            rows: 2,
            cols: 2,
            data: vec![-1.0, 0.0, 
                       2.0, 3.0],
        };
        let b = Matrix {
            rows: 2,
            cols: 2,
            data: vec![5.0, -2.0, 
                       0.0, 4.0],
        };

        let res = (&a * &b).unwrap();


        let expected_data = vec![-5.0, 2.0, 10.0, 8.0];
        
        assert_eq!(res.rows, 2);
        assert_eq!(res.cols, 2);
        assert_eq!(res.data, expected_data);
    }

    #[test]
    fn mul_row_vec_by_col_vec_dot_product() {
        let a = Matrix {
            rows: 1,
            cols: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        let b = Matrix {
            rows: 3,
            cols: 1,
            data: vec![4.0, 5.0, 6.0],
        };

        let res = (&a * &b).unwrap();
        
        let expected_data = vec![32.0];

        assert_eq!(res.rows, 1);
        assert_eq!(res.cols, 1);
        assert_eq!(res.data, expected_data);
    }

    #[test]
    fn mul_col_vec_by_row_vec_outer_product() {
        let a = Matrix {
            rows: 3,
            cols: 1,
            data: vec![1.0, 2.0, 3.0],
        };
        let b = Matrix {
            rows: 1,
            cols: 3,
            data: vec![4.0, 5.0, 6.0],
        };

        let res = (&a * &b).unwrap();

        let expected_data = vec![
            4.0, 5.0, 6.0,
            8.0, 10.0, 12.0,
            12.0, 15.0, 18.0,
        ];

        assert_eq!(res.rows, 3);
        assert_eq!(res.cols, 3);
        assert_eq!(res.data, expected_data);
    }

    #[test]
    #[should_panic]
    fn mul_mismatched_dimensions_should_panic() {
        let a = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0; 6],
        };
        let b = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0; 4],
        };

        let _res = (&a * &b).unwrap();
    }

    #[test]
    fn transpose_rectangular_matrix() {
        // A (2x3)
        let mut a = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 
                       4.0, 5.0, 6.0],
        };

        a.transpose();

        // A^T (3x2)
        let expected_data = vec![
            1.0, 4.0, 
            2.0, 5.0, 
            3.0, 6.0
        ];

        assert_eq!(a.rows, 3);
        assert_eq!(a.cols, 2);
        assert_eq!(a.data, expected_data);
    }

    #[test]
    fn transpose_col_vector_to_row_vector() {
        // A (3x1)
        let mut a = Matrix {
            rows: 3,
            cols: 1,
            data: vec![1.0, 2.0, 3.0],
        };

        a.transpose();

        // A^T (1x3)
        let expected_data = vec![1.0, 2.0, 3.0];

        assert_eq!(a.rows, 1);
        assert_eq!(a.cols, 3);
        assert_eq!(a.data, expected_data);
    }

    #[test]
    fn transpose_row_vector_to_col_vector() {
        // A (1x3)
        let mut a = Matrix {
            rows: 1,
            cols: 3,
            data: vec![1.0, 2.0, 3.0],
        };

        a.transpose();

        // A^T (3x1)
        let expected_data = vec![1.0, 2.0, 3.0];

        assert_eq!(a.rows, 3);
        assert_eq!(a.cols, 1);
        assert_eq!(a.data, expected_data);
    }
}