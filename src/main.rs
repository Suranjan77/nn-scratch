mod data;
mod math;

use crate::data::*;
use crate::math::matrix::Matrix;

fn main() {
    // Simple network

    // input: 1x1

    // hidden layer 1: 6x1
    let mut W1 = Matrix::uniform(6, 1);
    // hidden layer 2: 9x1
    let mut W2 = Matrix::uniform(9, 1);
    // hidden layer 3: 6x1
    let mut W3 = Matrix::uniform(6, 1);
    // output layer: 2x1
    let mut W4 = Matrix::uniform(2, 1);


}
