//! Parser for the IDX file format used by the MNIST dataset.
//!
//! IDX layout (all integers are big-endian):
//!
//! ```text
//! [0, 0, data_type, dim_count]   <- 4 byte "magic number"
//! [dim_0: u32]                   <- number of items (images or labels)
//! [dim_1: u32] ... [dim_n: u32]  <- size of each item (28 and 28 for images, nothing for labels)
//! [data ...]                     <- raw bytes, one u8 per value
//! ```
//!
//! For MNIST:
//! - label files have 1 dimension  -> every item is a single byte (the digit 0-9)
//! - image files have 3 dimensions -> every item is 28 * 28 = 784 bytes (pixels 0-255)

use crate::math::matrix::Matrix;
use std::{fs::File, io::BufReader, io::Read};

/// Reads an MNIST IDX file into ONE matrix with one row per item.
///
/// - Images become the design matrix X, shape (images, 784): one row per image,
///   one column per pixel. A digit is 784 numbers: 28 rows of 28 grey levels,
///   read row after row, with 0 for paper and 255 for ink. Pixels are scaled from
///   0..255 to 0.0..1.0. Scaling matters: with values up to 255 the weighted sums
///   become huge, the sigmoid gets stuck at 0 or 1 where its gradient is ~0, and the
///   network barely learns (or the loss blows up to infinity).
/// - Labels become the answer matrix Y, shape (images, 10), one-hot (see [`one_hot`]).
///
/// Row `i` of X and row `i` of Y belong to the same image.
pub fn parse(file_path: &str) -> Matrix {
    let file = File::open(file_path)
        .unwrap_or_else(|e| panic!("Failed to open file {}: {:?}", file_path, e));
    let mut br = BufReader::new(file);

    // Magic number. We know MNIST stores unsigned bytes, so we only care about
    // the last byte, which tells us how many dimensions follow.
    let mut magic_num = [0_u8; 4];
    br.read_exact(&mut magic_num).unwrap();
    let dim_count = magic_num[3] as usize;

    // Read ALL dimension sizes. Every dimension must be consumed from the header,
    // otherwise the data that follows would be read from the wrong offset.
    let mut dims = Vec::with_capacity(dim_count);
    for _ in 0..dim_count {
        let mut d = [0_u8; 4];
        br.read_exact(&mut d).unwrap();
        dims.push(u32::from_be_bytes(d) as usize);
    }

    // The first dimension is the number of items; the rest describe one item.
    // For labels `dims[1..]` is empty and the product of an empty list is 1.
    let item_count = dims[0];
    let item_size: usize = dims[1..].iter().product();

    // The rest of the file is just the bytes, item after item.
    let mut bytes = vec![0_u8; item_count * item_size];
    br.read_exact(&mut bytes).unwrap();

    if dim_count == 1 {
        one_hot(&bytes)
    } else {
        let pixels = bytes.iter().map(|&p| p as f64 / 255.0).collect();
        Matrix::new(item_count, item_size, pixels)
    }
}

/// Turns a list of digits into a one-hot matrix of shape (digits, 10).
///
/// One column per digit, with a 1 in the column of the true one:
///
/// ```text
/// label         0 1 2 3 4 5 6 7 8 9
///   5    ->    [0 0 0 0 0 1 0 0 0 0]
///   0    ->    [1 0 0 0 0 0 0 0 0 0]
///   4    ->    [0 0 0 0 1 0 0 0 0 0]
/// ```
///
/// The network predicts ten numbers per image, so the answer is ten numbers too,
/// and `prediction - answer` gives one error per image per digit.
fn one_hot(labels: &[u8]) -> Matrix {
    let mut data = vec![0.0; labels.len() * 10];
    for (i, &digit) in labels.iter().enumerate() {
        assert!(digit < 10, "MNIST labels must be between 0 and 9");
        data[i * 10 + digit as usize] = 1.0;
    }
    Matrix::new(labels.len(), 10, data)
}
