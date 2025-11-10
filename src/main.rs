mod data;
mod math;
mod nn;

use crate::data::{generator, idx_parser};
use crate::math::activation::{d_sigmoid, sigmoid, softmax};
use crate::math::loss_functions::{cross_entropy, sse};
use crate::math::matrix::Matrix;
use crate::nn::perceptron;
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufWriter, Read, Write};

fn main() {
    let mut train_data = idx_parser::parse(
        "C:\\Users\\e65455\\Documents\\git\\nn-scratch\\mnist_data\\train-images.idx3-ubyte",
    );
    let train_labels = idx_parser::parse(
        "C:\\Users\\e65455\\Documents\\git\\nn-scratch\\mnist_data\\train-labels.idx1-ubyte",
    );

    let mut nn = perceptron::Network::new(0.001, cross_entropy);
    nn.add_inp_layer(256, 784, sigmoid, Some(d_sigmoid));
    nn.add_layer(64, sigmoid, Some(d_sigmoid));
    nn.add_layer(10, softmax, None);

    let epoch = 100;
    let batch_size = train_data.len() / epoch;

    for e in 0..epoch {
        let train_batch_data = &mut train_data[e * batch_size..(e * batch_size + batch_size)];
        let train_batch_labels = &train_labels[e * batch_size..(e * batch_size + batch_size)];

        let mut training_loss = 0.0;

        for (i, x) in train_batch_data.iter_mut().enumerate() {
            let y = &train_batch_labels[i];
            training_loss += nn.feed_forward(x, y);
            nn.calc_gradients(x, y);
        }

        training_loss /= batch_size as f64;
        println!("Epoch: {} Train Loss: {}", e, training_loss);

        nn.update_gradients(batch_size);
    }

    // let test_data = idx_parser::parse("mnist_data/t10k-labels.idx3-ubyte");
    // let test_labels = idx_parser::parse("mnist_data/t10k-labels.idx3-ubyte");
}

#[allow(dead_code)]
fn train() {
    // True to the maths neural network, without optimisations

    let training_data = generator::generate_train(10000);
    // let validation_data = generator::generate_validation(300);

    // hidden layer 1: 6 neurons
    let mut w1 = Matrix::uniform(6, 1);
    let mut b1 = Matrix::repeat(6, 1, 1.0);

    // hidden layer 2: 12 neurons
    let mut w2 = Matrix::uniform(9, 6);
    let mut b2 = Matrix::repeat(9, 1, 1.0);

    // hidden layer 3: 6 neurons
    let mut w3 = Matrix::uniform(6, 9);
    let mut b3 = Matrix::repeat(6, 1, 1.0);

    // output layer: 2 neurons
    let mut w4 = Matrix::uniform(2, 6);
    let mut b4 = Matrix::repeat(2, 1, 1.0);

    let epoch = 100;
    let batch_size = 100;
    let learning_rate = 0.001;

    for e in 0..epoch {
        let train_batch = &training_data[e * batch_size..(e * batch_size + batch_size)];

        let mut training_loss = 0.0;

        let mut dw1 = Matrix::repeat(w1.rows(), w1.cols(), 0.0);
        let mut dw2 = Matrix::repeat(w2.rows(), w2.cols(), 0.0);
        let mut dw3 = Matrix::repeat(w3.rows(), w3.cols(), 0.0);
        let mut dw4 = Matrix::repeat(w4.rows(), w4.cols(), 0.0);

        let mut db1 = Matrix::repeat(b1.rows(), b1.cols(), 0.0);
        let mut db2 = Matrix::repeat(b2.rows(), b2.cols(), 0.0);
        let mut db3 = Matrix::repeat(b3.rows(), b3.cols(), 0.0);
        let mut db4 = Matrix::repeat(b4.rows(), b4.cols(), 0.0);

        for d in train_batch.iter() {
            let mut x = Matrix::new(1, 1, vec![d.0]);
            let y = Matrix::new(2, 1, d.1.to_vec());

            // ==== Feedforward ====
            // Layer 1
            let a1 = &w1.dot(&x).unwrap() + &b1;
            let mut z1 = sigmoid(&a1);

            // Layer 2
            let a2 = &w2.dot(&z1).unwrap() + &b2;
            let mut z2 = sigmoid(&a2);

            //Layer 3
            let a3 = &w3.dot(&z2).unwrap() + &b3;
            let mut z3 = sigmoid(&a3);

            // Layer 4
            let a4 = &w4.dot(&z3).unwrap() + &b4;
            let y_hat = softmax(&a4);
            training_loss += cross_entropy(&y_hat, &y);

            // ==== gradient calculation ====
            // layer 4 gradient
            let e4 = &y_hat - &y;
            z3.transpose();
            let grad_w4 = e4.dot(&z3).unwrap();

            // layer 3 gradient
            w4.transpose();
            let e3 = &w4.dot(&e4).unwrap() * &d_sigmoid(&a3);
            z2.transpose();
            let grad_w3 = e3.dot(&z2).unwrap();
            z2.transpose();

            // layer 2 gradient
            w3.transpose();
            let e2 = &w3.dot(&e3).unwrap() * &d_sigmoid(&a2);
            z1.transpose();
            let grad_w2 = e2.dot(&z1).unwrap();
            z1.transpose();

            // layer 1 gradient
            w2.transpose();
            let e1 = &w2.dot(&e2).unwrap() * &d_sigmoid(&a1);
            x.transpose();
            let grad_w1 = e1.dot(&x).unwrap();
            x.transpose();

            dw1 = &dw1 + &grad_w1;
            dw2 = &dw2 + &grad_w2;
            dw3 = &dw3 + &grad_w3;
            dw4 = &dw4 + &grad_w4;

            db1 = &db1 + &e1;
            db2 = &db2 + &e2;
            db3 = &db3 + &e3;
            db4 = &db4 + &e4;

            w2.transpose();
            w3.transpose();
            w4.transpose();
        }

        // Update weights and bias
        w1 = &w1
            - &(&Matrix::repeat(w1.rows(), w1.cols(), learning_rate / batch_size as f64) * &dw1);
        w2 = &w2
            - &(&Matrix::repeat(w2.rows(), w2.cols(), learning_rate / batch_size as f64) * &dw2);
        w3 = &w3
            - &(&Matrix::repeat(w3.rows(), w3.cols(), learning_rate / batch_size as f64) * &dw3);
        w4 = &w4
            - &(&Matrix::repeat(w4.rows(), w4.cols(), learning_rate / batch_size as f64) * &dw4);

        b1 = &b1
            - &(&Matrix::repeat(b1.rows(), b1.cols(), learning_rate / batch_size as f64) * &db1);
        b2 = &b2
            - &(&Matrix::repeat(b2.rows(), b2.cols(), learning_rate / batch_size as f64) * &db2);
        b3 = &b3
            - &(&Matrix::repeat(b3.rows(), b3.cols(), learning_rate / batch_size as f64) * &db3);
        b4 = &b4
            - &(&Matrix::repeat(b4.rows(), b4.cols(), learning_rate / batch_size as f64) * &db4);

        training_loss /= batch_size as f64;
        println!("Epoch: {} Train Loss: {}", e, training_loss);
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

#[allow(dead_code)]
fn test() {
    let test_data = generator::generate_test(10);

    let mut file = File::open("model.json").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    let model: Value = serde_json::from_str(&contents).unwrap();

    let w1: Matrix = serde_json::from_value(model["weights"][0].clone()).unwrap();
    let w2: Matrix = serde_json::from_value(model["weights"][1].clone()).unwrap();
    let w3: Matrix = serde_json::from_value(model["weights"][2].clone()).unwrap();
    let w4: Matrix = serde_json::from_value(model["weights"][3].clone()).unwrap();

    let b1: Matrix = serde_json::from_value(model["biases"][0].clone()).unwrap();
    let b2: Matrix = serde_json::from_value(model["biases"][1].clone()).unwrap();
    let b3: Matrix = serde_json::from_value(model["biases"][2].clone()).unwrap();
    let b4: Matrix = serde_json::from_value(model["biases"][3].clone()).unwrap();

    let mut test_loss = 0.0;

    for d in test_data.iter() {
        let x = Matrix::new(1, 1, vec![d.0]);
        let y = Matrix::new(2, 1, d.1.to_vec());

        // ==== Feedforward ====
        // Layer 1
        let a1 = &w1.dot(&x).unwrap() + &b1;
        let z1 = sigmoid(&a1);

        // Layer 2
        let a2 = &w2.dot(&z1).unwrap() + &b2;
        let z2 = sigmoid(&a2);

        //Layer 3
        let a3 = &w3.dot(&z2).unwrap() + &b3;
        let z3 = sigmoid(&a3);

        // Layer 4
        let a4 = &w4.dot(&z3).unwrap() + &b4;
        let yhat = sigmoid(&a4);

        println!("X: {} Y: {}", x, yhat);
        test_loss += sse(&yhat, &y);
    }

    test_loss /= test_data.len() as f64 * 2.0;
    println!("Test Loss: {}", test_loss);
}
