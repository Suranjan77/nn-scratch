mod data;
mod math;

use crate::data::generator;
use crate::math::activation::{d_sigmoid, sigmoid};
use crate::math::loss_functions::sse;
use crate::math::matrix::Matrix;
use serde_json::json;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
   train()
}

fn train() {
    // True to the maths neural network, without optimisations

    let training_data = generator::generate_train(1000);
    // let validation_data = generator::generate_validation(300);

    // hidden layer 1: 6 neurons
    let mut w1 = Matrix::uniform(1, 6);
    let mut b1 = Matrix::repeat(6, 1, 1.0);

    // hidden layer 2: 9 neurons
    let mut w2 = Matrix::uniform(6, 9);
    let mut b2 = Matrix::repeat(9, 1, 1.0);

    // hidden layer 3: 6 neurons
    let mut w3 = Matrix::uniform(9, 6);
    let mut b3 = Matrix::repeat(6, 1, 1.0);

    // output layer: 2 neurons
    let mut w4 = Matrix::uniform(6, 2);
    let mut b4 = Matrix::repeat(2, 1, 1.0);

    let epoch = 5;
    let batch_size = 1000 / 5;
    let learning_rate = 0.01;

    for e in 0..epoch {
        let train_batch = &training_data[e * batch_size..(e * batch_size + batch_size)];

        let mut training_loss = 0.0;

        let mut grad_w1s: Vec<Matrix> = vec![];
        let mut grad_w2s: Vec<Matrix> = vec![];
        let mut grad_w3s: Vec<Matrix> = vec![];
        let mut grad_w4s: Vec<Matrix> = vec![];
        let mut grad_b1s: Vec<Matrix> = vec![];
        let mut grad_b2s: Vec<Matrix> = vec![];
        let mut grad_b3s: Vec<Matrix> = vec![];
        let mut grad_b4s: Vec<Matrix> = vec![];

        for d in train_batch.iter() {
            let mut x = Matrix::new(1, 1, vec![d.0]);
            let y = Matrix::new(2, 1, d.1.to_vec());

            // ==== Feedforward ====
            // Layer 1
            w1.transpose();
            let a1 = &w1.dot(&x).unwrap() + &b1;
            let mut z1 = sigmoid(&a1);

            // Layer 2
            w2.transpose();
            let a2 = &w2.dot(&z1).unwrap() + &b2;
            let mut z2 = sigmoid(&a2);

            //Layer 3
            w3.transpose();
            let a3 = &w3.dot(&z2).unwrap() + &b3;
            let mut z3 = sigmoid(&a3);

            // Layer 4
            w4.transpose();
            let a4 = &w4.dot(&z3).unwrap() + &b4;
            let yhat = sigmoid(&a4);
            training_loss += sse(&yhat, &y);

            // ==== gradient calculation ====
            // layer 4 gradient
            let e4 = &(&yhat - &y) * &d_sigmoid(&a4);
            z3.transpose();
            let grad_w4 = e4.dot(&z3).unwrap();
            z3.transpose();

            // layer 3 gradient
            //todo:Issue here with shape of w4 and e4, w4 => 2x6 cuz transpose for layer 4 and e4=>2x1
            let e3 = &w4.dot(&e4).unwrap() * &d_sigmoid(&a3);
            z2.transpose();
            let grad_w3 = e3.dot(&z2).unwrap();
            z2.transpose();

            // layer 2 gradient
            let e2 = &w3.dot(&e3).unwrap() * &d_sigmoid(&a2);
            z1.transpose();
            let grad_w2 = e2.dot(&z1).unwrap();
            z1.transpose();

            // layer 1 gradient
            let e1 = &w2.dot(&e2).unwrap() * &d_sigmoid(&a1);
            x.transpose();
            let grad_w1 = e1.dot(&x).unwrap();
            x.transpose();

            grad_w4s.push(grad_w4);
            grad_w3s.push(grad_w3);
            grad_w2s.push(grad_w2);
            grad_w1s.push(grad_w1);

            grad_b4s.push(e4);
            grad_b3s.push(e3);
            grad_b2s.push(e2);
            grad_b1s.push(e1);

            // Transpose weights back to initial shape
            w1.transpose();
            w2.transpose();
            w3.transpose();
            w4.transpose();
        }

        // Update weights and bias
        // calculating average of individual training gradient multiplied by learning rate
        let dw1 = &Matrix::repeat(w1.rows(), w1.cols(), learning_rate / batch_size as f64)
            * &grad_w1s.iter().fold(
                Matrix::repeat(w1.rows(), w1.cols(), 0.0),
                |acc: Matrix, e| &acc + e,
            );
        let dw2 = &Matrix::repeat(w2.rows(), w2.cols(), learning_rate / batch_size as f64)
            * &grad_w2s.iter().fold(
                Matrix::repeat(w2.rows(), w2.cols(), 0.0),
                |acc: Matrix, e| &acc + e,
            );
        let dw3 = &Matrix::repeat(w3.rows(), w3.cols(), learning_rate / batch_size as f64)
            * &grad_w3s.iter().fold(
                Matrix::repeat(w3.rows(), w3.cols(), 0.0),
                |acc: Matrix, e| &acc + e,
            );
        let dw4 = &Matrix::repeat(w4.rows(), w4.cols(), learning_rate / batch_size as f64)
            * &grad_w4s.iter().fold(
                Matrix::repeat(w4.rows(), w4.cols(), 0.0),
                |acc: Matrix, e| &acc + e,
            );

        let db1 = &Matrix::repeat(b1.rows(), b1.cols(), learning_rate / batch_size as f64)
            * &grad_b1s.iter().fold(
                Matrix::repeat(b1.rows(), b1.cols(), 0.0),
                |acc: Matrix, e| &acc + e,
            );
        let db2 = &Matrix::repeat(b2.rows(), b2.cols(), learning_rate / batch_size as f64)
            * &grad_b2s.iter().fold(
                Matrix::repeat(b2.rows(), b2.cols(), 0.0),
                |acc: Matrix, e| &acc + e,
            );
        let db3 = &Matrix::repeat(b3.rows(), b3.cols(), learning_rate / batch_size as f64)
            * &grad_b3s.iter().fold(
                Matrix::repeat(b3.rows(), b3.cols(), 0.0),
                |acc: Matrix, e| &acc + e,
            );
        let db4 = &Matrix::repeat(b4.rows(), b4.cols(), learning_rate / batch_size as f64)
            * &grad_b4s.iter().fold(
                Matrix::repeat(b4.rows(), b4.cols(), 0.0),
                |acc: Matrix, e| &acc + e,
            );

        // Back propagate gradients
        w1 = &w1 - &dw1;
        w2 = &w2 - &dw2;
        w3 = &w3 - &dw3;
        w4 = &w4 - &dw4;

        b1 = &b1 - &db1;
        b2 = &b2 - &db2;
        b3 = &b3 - &db3;
        b4 = &b4 - &db4;

        training_loss /= batch_size as f64 * 2.0;
        println!("Epoch: {} Train Loss: {}", epoch, training_loss);
    }

    match File::create("model.json") {
        Ok(m_file) => {
            let mut writer = BufWriter::new(m_file);
            let nn_model = json!({
                "weights": [w1, w2, w3, w4],
                "biases": [b1, b2, b3, b4],
                "activation": "sigmoid",
            });

            let json_string = serde_json::to_string(&nn_model).unwrap();
            writer.write_all(json_string.as_bytes()).unwrap();
        }
        Err(e) => {
            println!("Error opening file: {}", e);
        }
    }
}
