//! Trains a small multilayer perceptron to read handwritten digits (MNIST).
//!
//! Learning is one loop of matrix arithmetic:
//!   multiply, add, compare with the answer, correct the weights, repeat.
//!
//! Network: 784 pixels -> 256 features -> 64 features -> 10 scores
//!   - the hidden layers turn pixels into features (is this stroke here? is that loop closed?)
//!   - the last layer scores each digit 0-9 from those features
//!   - softmax turns each image's 10 scores into 10 probabilities that add up to 1
//!
//! Run with `cargo run --release` (debug builds are much slower).

mod data;
mod math;
mod nn;

use rand::seq::SliceRandom;

use crate::data::idx_parser;
use crate::math::activation::{d_sigmoid, sigmoid, softmax};
use crate::math::loss_functions::cross_entropy;
use crate::math::matrix::Matrix;
use crate::nn::perceptron::{self, Network};

fn main() {
    // ---- Load data ----------------------------------------------------------
    // The dataset is a matrix. X is the design matrix: one row per image, one
    // column per pixel, (60000, 784), scaled to 0..1. Y holds the answers in the
    // same order, one-hot: (60000, 10). Row i of X goes with row i of Y.
    let mut x_train = idx_parser::parse("./mnist_data/train-images.idx3-ubyte");
    let mut y_train = idx_parser::parse("./mnist_data/train-labels.idx1-ubyte");
    let x_test = idx_parser::parse("./mnist_data/t10k-images.idx3-ubyte");
    let y_test = idx_parser::parse("./mnist_data/t10k-labels.idx1-ubyte");

    // Three sets, three jobs:
    //   training   (50 000) - the weights learn from it
    //   validation (10 000) - we watch it while training to choose settings
    //                         (learning rate, layer sizes, number of epochs)
    //   test       (10 000) - scored ONCE at the very end and reported.
    // If we chose settings by looking at the test score, we would be fitting the
    // test set and its score would no longer tell us how new digits will do.
    // `split_off_rows` moves the last 10 000 rows into a new matrix.
    let x_val = x_train.split_off_rows(50_000);
    let y_val = y_train.split_off_rows(50_000);
    println!(
        "X: ({}, {})  Y: ({}, {})",
        x_train.rows, x_train.cols, y_train.rows, y_train.cols
    );

    // ---- Build the network --------------------------------------------------
    // Learning rate 1.0: how big a step each weight takes against its slope.
    // Too small and learning crawls, too large and the loss jumps around.
    let mut nn = perceptron::Network::new(1.0, cross_entropy);
    nn.add_inp_layer(256, 784, sigmoid, Some(d_sigmoid)); // W1: (784, 256)
    nn.add_layer(64, sigmoid, Some(d_sigmoid)); //           W2: (256, 64)
    nn.add_layer(10, softmax, None); //                      W3: (64, 10)

    // ---- Train ---------------------------------------------------------------
    // Minibatches: the same arithmetic on 100 rows at a time. Shuffle, cut into
    // batches of 100, update after each: 500 updates per epoch.
    // One epoch = every training image seen once.
    let epochs = 5;
    let batch_size = 100;

    // We shuffle row *indices*, not the data: one index picks the image (row of X)
    // and its answer (row of Y) together, so they stay paired.
    let mut order: Vec<usize> = (0..x_train.rows).collect();

    for epoch in 0..epochs {
        // A new order every epoch, so every batch is a different random mix of digits.
        order.shuffle(&mut rand::rng());

        let mut training_loss = 0.0;

        for rows in order.chunks(batch_size) {
            let xb = x_train.select_rows(rows); // (100, 784)
            let yb = y_train.select_rows(rows); // (100, 10)

            // forward, compare, backward, update - for the whole batch at once.
            // The step returns the batch's average loss; weigh it by the batch size
            // because the last batch may be smaller.
            training_loss += nn.train_step(&xb, &yb) * rows.len() as f64;
        }

        // The training loss should fall every epoch. If the validation accuracy
        // stops improving while the training loss keeps falling, the network has
        // started memorising the training digits (overfitting).
        training_loss /= x_train.rows as f64;
        println!(
            "Epoch: {} Train Loss: {:.4} Validation Accuracy: {:.2}%",
            epoch + 1,
            training_loss,
            accuracy(&nn, &x_val, &y_val) * 100.0
        );
    }

    // ---- Report --------------------------------------------------------------
    // Touch the test set once. For comparison: always answering "1" (the most
    // common digit) is already right 11.35% of the time, so a score only means
    // something when it clearly beats that baseline.
    println!(
        "Test Accuracy: {:.2}%",
        accuracy(&nn, &x_test, &y_test) * 100.0
    );

    nn.save("model.json");
    println!("Model saved to model.json");
}

/// Fraction (0.0 - 1.0) of images the network reads correctly.
///
/// All images go through the network in one go: P has one row per image.
/// The predicted digit is the column with the highest probability in each row
/// of P, and the true digit is the column holding the 1 in each row of Y.
fn accuracy(nn: &Network, x: &Matrix, y: &Matrix) -> f64 {
    let predicted = nn.predict(x).argmax_rows();
    let actual = y.argmax_rows();

    let correct = predicted
        .iter()
        .zip(&actual)
        .filter(|(p, a)| p == a)
        .count();

    correct as f64 / actual.len() as f64
}
