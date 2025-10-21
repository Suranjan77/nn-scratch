
#[derive(Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

use std::ops::Mul;

impl<'a> Mul for &'a Matrix {
    type Output = Matrix;

    // Naive implementation with 3 loops. It needs optimisation
    fn mul(self, other: &'a Matrix) -> Matrix {
        let res_len = self.rows * other.cols;
        let mut res = Matrix {
            rows: self.rows,
            cols: other.cols,
            data: vec![0.0; res_len],
        };

        for i in 0..self.rows { // [2, 3, 4, 3, 1, 2, 4, 2, 5]
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                res.data[i * other.cols + j] = sum;
            }
        }

        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_by_tow() {
        let a = Matrix {rows: 2, cols: 2, data: vec![2.1, 2.3, 1.2, 1.4]};
        let b = Matrix {rows: 2, cols: 2, data: vec![1.3, 1.2, 2.9, 2.7]};

        let res = &a * &b;

        assert_eq!(res.rows, a.rows);
        assert_eq!(res.cols, b.cols);
        assert_eq!(res.data, vec![9.399999999999999, 8.73, 5.619999999999999, 5.22]);
    }

    #[test]
    fn three_by_one() {
        let a = Matrix {rows: 3, cols: 3, data: vec![2.0;9]};
        let b = Matrix {rows: 3, cols: 1, data: vec![1.0, 2.0, 3.0]};

        let res = &a * &b;

        assert_eq!(res.rows, a.rows);
        assert_eq!(res.cols, b.cols);
        assert_eq!(res.data, vec![12.0, 12.0, 12.0]);
    }

    #[test]
    fn three_by_three() {
        let a = Matrix {rows: 3, cols: 3, data: vec![2.0;9]};
        let b = Matrix {rows: 3, cols: 3, data: vec![2.0;9]};

        let res = &a * &b;

        assert_eq!(res.rows, a.rows);
        assert_eq!(res.cols, b.cols);
        assert_eq!(res.data, vec![12.0;9]);
    }
}