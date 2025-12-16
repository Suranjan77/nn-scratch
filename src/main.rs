mod data;
mod math;
mod nn;

use crate::data::idx_parser;
use crate::math::activation::{d_sigmoid, sigmoid, softmax};
use crate::math::loss_functions::cross_entropy;
use crate::nn::perceptron;

fn main() {
    let mut train_data = idx_parser::parse(
        "./mnist_data/train-images.idx3-ubyte",
    );
    let train_labels = idx_parser::parse(
        "./mnist_data/train-labels.idx1-ubyte",
    );

    let mut nn = perceptron::Network::new(0.001, cross_entropy);
    nn.add_inp_layer(256, 784, sigmoid, Some(d_sigmoid));
    nn.add_layer(128, sigmoid, Some(d_sigmoid));
    nn.add_layer(64, sigmoid, Some(d_sigmoid));
    nn.add_layer(10, softmax, None);

    let epoch = 50;
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
}
