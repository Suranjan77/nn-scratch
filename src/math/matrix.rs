//! A minimal 2D matrix of `f64` values stored in a single flat `Vec`.
//!
//! Element `(row, col)` lives at index `row * row_stride + col * col_stride`.
//! For a normal (row-major) matrix `row_stride = cols` and `col_stride = 1`.
//! Keeping strides lets `transpose()` be free: it just swaps the dimensions and
//! the strides instead of moving any data around.

use rand::distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::ops::{Add, Mul, Sub};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    row_stride: usize,
    col_stride: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Creates a `rows x cols` matrix from row-major `data`
    /// (the first row, then the second row, ...).
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        if rows * cols != data.len() {
            panic!("Data length does not match dimensions");
        }
        Self {
            data,
            rows,
            cols,
            col_stride: 1,
            row_stride: cols,
        }
    }

    /// Raw underlying data. Note: this is the storage order, which is NOT the
    /// logical order for a transposed matrix. Use `get` when that matters.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Reads element `(row, col)` taking the strides into account.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.row_stride + col * self.col_stride]
    }

    /// Matrix filled with random values in `[-1, 1)`.
    pub fn uniform(rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();
        let distribution = Uniform::try_from(-1.0..1.0).unwrap();
        let data = (0..rows * cols)
            .map(|_| distribution.sample(&mut rng))
            .collect();
        Matrix::new(rows, cols, data)
    }

    /// Identity matrix of size `size x size`.
    pub fn eye(size: usize) -> Self {
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }
        Matrix::new(size, size, data)
    }

    /// Matrix where every element is `repeat_value`.
    pub fn repeat(rows: usize, cols: usize, repeat_value: f64) -> Self {
        let data = vec![repeat_value; rows * cols];
        Matrix::new(rows, cols, data)
    }

    /// Transposes in place by swapping dimensions and strides (no data is copied).
    pub fn transpose(&mut self) {
        std::mem::swap(&mut self.rows, &mut self.cols);
        std::mem::swap(&mut self.row_stride, &mut self.col_stride);
    }

    /// Returns a transposed copy, `self^T`, leaving `self` unchanged.
    /// Rows become columns: a (100, 784) matrix becomes (784, 100).
    pub fn transposed(&self) -> Matrix {
        let mut t = self.clone();
        t.transpose();
        t
    }

    /// Matrix product `self · other`: row `i` of `self` times column `k` of `other`
    /// gives element `(i, k)` of the result. Shapes: (n, m) · (m, p) -> (n, p).
    pub fn dot(&self, other: &Matrix) -> Result<Matrix, &'static str> {
        if self.cols != other.rows {
            return Err("Matrix multiplication dimension mismatch: A.cols != B.rows");
        }

        let mut data = vec![0.0; self.rows * other.cols];

        // Matrix multiplication using double precision matrices multiplication routine from matrixmultiply crate.
        // It accepts strides, so transposed matrices work without copying.
        unsafe {
            matrixmultiply::dgemm(
                self.rows,
                self.cols,
                other.cols,
                1.0,
                self.data.as_ptr(),
                self.row_stride as isize,
                self.col_stride as isize,
                other.data.as_ptr(),
                other.row_stride as isize,
                other.col_stride as isize,
                0.0,
                data.as_mut_ptr(),
                other.cols as isize,
                1,
            );
        }

        Ok(Matrix::new(self.rows, other.cols, data))
    }

    /// Raises every element to the power `exp`.
    pub fn powi(&self, exp: i32) -> Self {
        let mut data = vec![0.0; self.cols * self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[i * self.cols + j] = self.get(i, j).powi(exp);
            }
        }
        Matrix::new(self.rows, self.cols, data)
    }

    /// Multiplies every element by `factor` and returns the result as a new matrix.
    pub fn scale(&self, factor: f64) -> Self {
        let mut data = vec![0.0; self.cols * self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[i * self.cols + j] = self.get(i, j) * factor;
            }
        }
        Matrix::new(self.rows, self.cols, data)
    }

    /// Adds up each column: (n, cols) -> (1, cols).
    /// Used for the bias gradient, where the errors of all images in a batch
    /// are added together, one total per neuron.
    pub fn sum_columns(&self) -> Matrix {
        let mut data = vec![0.0; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j] += self.get(i, j);
            }
        }
        Matrix::new(1, self.cols, data)
    }

    /// For every row, the column index of its largest value.
    /// For a matrix of predictions (one row per image, one column per digit)
    /// this is the predicted digit of every image.
    pub fn argmax_rows(&self) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut best = 0;
            for j in 1..self.cols {
                if self.get(i, j) > self.get(i, best) {
                    best = j;
                }
            }
            result.push(best);
        }
        result
    }

    /// Builds a new matrix from the given rows, in the given order.
    /// Used to cut a minibatch out of the design matrix: the same row indices
    /// picked from X and from Y keep every image paired with its answer.
    pub fn select_rows(&self, rows: &[usize]) -> Matrix {
        let mut data = Vec::with_capacity(rows.len() * self.cols);
        for &i in rows {
            for j in 0..self.cols {
                data.push(self.get(i, j));
            }
        }
        Matrix::new(rows.len(), self.cols, data)
    }

    /// Splits the matrix in two: `self` keeps rows `0..at`, and rows `at..` are
    /// moved into the returned matrix. No data is copied.
    pub fn split_off_rows(&mut self, at: usize) -> Matrix {
        assert!(
            self.col_stride == 1 && self.row_stride == self.cols,
            "split_off_rows needs a matrix that is not transposed"
        );
        let rest = self.data.split_off(at * self.cols);
        let rest_rows = self.rows - at;
        self.rows = at;
        Matrix::new(rest_rows, self.cols, rest)
    }

    /// Applies `op` element-wise to `self` and `other`.
    ///
    /// Supported shapes for `other`:
    /// - 1x1         -> treated as a scalar applied to every element
    /// - 1 x cols    -> the single row is broadcast to (repeated for) every row of `self`;
    ///                  this is how one bias row is added to every image of a batch
    /// - rows x cols -> plain element-wise operation
    fn zip_map<F>(&self, other: &Matrix, op: F) -> Matrix
    where
        F: Fn(f64, f64) -> f64,
    {
        // Scalar case
        if other.data.len() == 1 {
            let scalar = other.data[0];
            let data = self.data.iter().map(|&x| op(x, scalar)).collect();
            return Matrix::new(self.rows, self.cols, data);
        }

        // Dimension Check
        if self.cols != other.cols {
            panic!("Dimensions are not compatible: cols mismatch");
        }

        let output_rows = self.rows;
        let output_cols = self.cols;
        let mut new_data = Vec::with_capacity(output_rows * output_cols);

        // Row Broadcasting logic
        let broadcast_rows = other.rows == 1;
        if self.rows != other.rows && !broadcast_rows {
            panic!("Dimensions are not compatible: rows mismatch");
        }

        // Iterate logically (i, j) and access physical data via strides
        for i in 0..output_rows {
            for j in 0..output_cols {
                let val_self = self.get(i, j);

                // If broadcasting B (1 row), always use row 0
                let row_other = if broadcast_rows { 0 } else { i };
                let val_other = other.get(row_other, j);

                new_data.push(op(val_self, val_other));
            }
        }

        Matrix::new(output_rows, output_cols, new_data)
    }
}

/// Element-wise (Hadamard) product: `&a * &b`. Not the matrix product, use `dot` for that.
impl Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Matrix {
        self.zip_map(other, |a, b| a * b)
    }
}

/// Element-wise addition: `&a + &b`.
impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Matrix {
        self.zip_map(other, |a, b| a + b)
    }
}

/// Element-wise subtraction: `&a - &b`.
impl Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, other: &Matrix) -> Matrix {
        self.zip_map(other, |a, b| a - b)
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut disp = String::with_capacity(2 * (self.rows * self.cols));
        disp.push('\n');
        for i in 0..self.rows {
            for j in 0..self.cols {
                disp.push_str(&format!("{} ", self.get(i, j)));
            }
            disp.push('\n');
        }
        disp.push_str(&format!("({}, {})\n", self.rows, self.cols));
        write!(f, "{}", disp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_transpose_then_add() {
        // A = [1, 2] (1x2)
        let mut a = Matrix::new(1, 2, vec![1.0, 2.0]);

        // Transpose A -> [1, 2]^T (2x1)
        a.transpose();

        // B = [3, 4]^T (2x1)
        let b = Matrix::new(2, 1, vec![3.0, 4.0]);

        // Operation:
        // [1]   [3]   [4]
        // [2] + [4] = [6]
        let res = &a + &b;

        assert_eq!(res.rows, 2);
        assert_eq!(res.cols, 1);
        assert_eq!(res.data, vec![4.0, 6.0]);
    }

    #[test]
    fn two_by_tow() {
        let a = Matrix::new(2, 2, vec![2.1, 2.3, 1.2, 1.4]);
        let b = Matrix::new(2, 2, vec![1.3, 1.2, 2.9, 2.7]);

        let res = &a.dot(&b).unwrap();

        assert_eq!(res.rows, a.rows);
        assert_eq!(res.cols, b.cols);
        assert_vec_approx_eq(
            &res.data,
            &[9.399999999999999, 8.73, 5.619999999999999, 5.22],
        );
    }

    #[test]
    fn three_by_one() {
        let a = Matrix::repeat(3, 3, 2.0);
        let b = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);

        let res = &a.dot(&b).unwrap();

        assert_eq!(res.rows, a.rows);
        assert_eq!(res.cols, b.cols);
        assert_eq!(res.data, vec![12.0, 12.0, 12.0]);
    }

    #[test]
    fn three_by_three() {
        let a = Matrix::repeat(3, 3, 2.0);
        let b = Matrix::repeat(3, 3, 2.0);

        let res = &a.dot(&b).unwrap();

        assert_eq!(res.rows, a.rows);
        assert_eq!(res.cols, b.cols);
        assert_eq!(res.data, vec![12.0; 9]);
    }

    #[test]
    fn four_by_four() {
        let a = Matrix::repeat(4, 4, 2.0);
        let b = Matrix::repeat(4, 4, 2.0);

        let res = &a.dot(&b).unwrap();

        assert_eq!(res.rows, a.rows);
        assert_eq!(res.cols, b.cols);
        assert_eq!(res.data, vec![16.0; 16]); // 4 * (2.0 * 2.0) = 16.0
    }

    #[test]
    fn one_twentyone_by_one_twentyone() {
        let a = Matrix::repeat(121, 121, 2.0);
        let b = Matrix::repeat(121, 121, 2.0);

        let res = &a.dot(&b).unwrap();

        assert_eq!(res.rows, a.rows);
        assert_eq!(res.cols, b.cols);
        assert_eq!(res.data, vec![484.0; 14641]); // 121 * (2.0 * 2.0) = 484.0
    }

    #[test]
    fn mul_rectangular_matrices() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let res = &a.dot(&b).unwrap();

        let expected_data = vec![58.0, 64.0, 139.0, 154.0];

        assert_eq!(res.rows, 2);
        assert_eq!(res.cols, 2);
        assert_eq!(res.data, expected_data);
    }

    #[test]
    fn mul_identity_matrices() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let i_3x3 = Matrix::new(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let i_2x2 = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);

        // Test A * I = A
        let res_a = &a.dot(&i_3x3).unwrap();
        assert_eq!(res_a.rows, a.rows);
        assert_eq!(res_a.cols, a.cols);
        assert_eq!(res_a.data, a.data);

        // Test I * A = A
        let res_b = &i_2x2.dot(&a).unwrap();
        assert_eq!(res_b.rows, a.rows);
        assert_eq!(res_b.cols, a.cols);
        assert_eq!(res_b.data, a.data);
    }

    #[test]
    fn mul_with_zeros_and_negatives() {
        let a = Matrix::new(2, 2, vec![-1.0, 0.0, 2.0, 3.0]);
        let b = Matrix::new(2, 2, vec![5.0, -2.0, 0.0, 4.0]);

        let res = &a.dot(&b).unwrap();

        let expected_data = vec![-5.0, 2.0, 10.0, 8.0];

        assert_eq!(res.rows, 2);
        assert_eq!(res.cols, 2);
        assert_eq!(res.data, expected_data);
    }

    #[test]
    fn mul_row_vec_by_col_vec_dot_product() {
        let a = Matrix::new(1, 3, vec![1., 2., 3.]);
        let b = Matrix::new(3, 1, vec![4., 5., 6.]);

        let res = &a.dot(&b).unwrap();

        let expected_data = vec![32.0];

        assert_eq!(res.rows, 1);
        assert_eq!(res.cols, 1);
        assert_eq!(res.data, expected_data);
    }

    #[test]
    fn mul_col_vec_by_row_vec_outer_product() {
        let a = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);
        let b = Matrix::new(1, 3, vec![4.0, 5.0, 6.0]);

        let res = &a.dot(&b).unwrap();

        let expected_data = vec![4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0];

        assert_eq!(res.rows, 3);
        assert_eq!(res.cols, 3);
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

        assert_eq!(a.rows, 3);
        assert_eq!(a.cols, 2);
        assert_eq!(a.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn transpose_col_vector_to_row_vector() {
        // A (3x1)
        let mut a = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);

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
        let mut a = Matrix::new(1, 3, vec![1., 2., 3.]);

        a.transpose();

        // A^T (3x1)
        let expected_data = vec![1.0, 2.0, 3.0];

        assert_eq!(a.rows, 3);
        assert_eq!(a.cols, 1);
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
