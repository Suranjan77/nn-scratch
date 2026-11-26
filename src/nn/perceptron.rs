//! A simple fully-connected feed-forward neural network (multilayer perceptron).
//!
//! Everything works on whole batches at once, using the design matrix:
//! one ROW per image, one COLUMN per pixel (or per feature, or per digit).
//!
//! A multilayer perceptron is the same layer several times: the output of one
//! layer is the input of the next. Each layer does
//!
//! ```text
//! A = X · W + b     multiply and add: row i, column k is image i's score for neuron k
//! Z = f(A)          activation function, applied to every value on its own
//! ```
//!
//! Shapes, for a batch of `n` images going through a layer with `inputs` inputs
//! and `neurons` outputs:
//!
//! ```text
//! X : (n, inputs)        the images (or the previous layer's output Z)
//! W : (inputs, neurons)  one COLUMN of weights per neuron, one weight per input
//! b : (1, neurons)       one bias per neuron, added to every row (broadcast)
//! A, Z : (n, neurons)
//! ```
//!
//! For our network: X (n, 784) -> Z1 (n, 256) -> Z2 (n, 64) -> P (n, 10).
//!
//! Training is one step repeated for every minibatch (`train_step`):
//! forward, compare with the answer, send the error back, update the weights.

use crate::math::matrix::Matrix;

/// One fully-connected layer.
struct Layer {
    weights: Matrix,
    bias: Matrix,
    activation: fn(&Matrix) -> Matrix,
    /// Derivative of `activation`. `None` for the softmax output layer, because the
    /// error there is simply `P - Y` (see `train_step`).
    d_activation: Option<fn(&Matrix) -> Matrix>,
}

pub struct Network {
    layers: Vec<Layer>,
    learning_rate: f64,
    loss_fn: fn(&Matrix, &Matrix) -> f64,
}

impl Network {
    pub fn new(learning_rate: f64, loss_fn: fn(&Matrix, &Matrix) -> f64) -> Network {
        Network {
            learning_rate,
            loss_fn,
            layers: vec![],
        }
    }

    /// Adds the first layer, which reads the raw input of size `input_size`.
    pub fn add_inp_layer(
        &mut self,
        neurons: usize,
        input_size: usize,
        activation: fn(&Matrix) -> Matrix,
        d_activation: Option<fn(&Matrix) -> Matrix>,
    ) {
        self.push_layer(input_size, neurons, activation, d_activation);
    }

    /// Adds a layer whose input is the output of the previously added layer.
    pub fn add_layer(
        &mut self,
        neurons: usize,
        activation: fn(&Matrix) -> Matrix,
        d_activation: Option<fn(&Matrix) -> Matrix>,
    ) {
        if self.layers.is_empty() {
            panic!("Add an input layer before adding hidden layers");
        }

        // The previous layer has one output per column of its weight matrix.
        let inputs = self.layers.last().unwrap().weights.cols;
        self.push_layer(inputs, neurons, activation, d_activation);
    }

    /// Creates a layer with `inputs` inputs and `neurons` outputs.
    fn push_layer(
        &mut self,
        inputs: usize,
        neurons: usize,
        activation: fn(&Matrix) -> Matrix,
        d_activation: Option<fn(&Matrix) -> Matrix>,
    ) {
        // Starting weights are random, so that neurons start different from each
        // other and can learn different features. They are scaled by 1/sqrt(inputs):
        // a neuron summing 784 inputs would otherwise start with huge scores, pushing
        // the sigmoid into its flat ends where the gradient is ~0 and nothing is learned.
        let weights = Matrix::uniform(inputs, neurons).scale(1.0 / (inputs as f64).sqrt());

        self.layers.push(Layer {
            weights,
            bias: Matrix::repeat(1, neurons, 0.0),
            activation,
            d_activation,
        });
    }

    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Forward pass for PREDICTION: every image of `x` through every layer.
    /// Returns P, shape (images, 10): one row of ten probabilities per image.
    pub fn predict(&self, x: &Matrix) -> Matrix {
        let mut z = x.clone();

        for layer in &self.layers {
            // The output of one layer is the input of the next.
            let a = &z.dot(&layer.weights).unwrap() + &layer.bias; // A = Z·W + b
            z = (layer.activation)(&a); // Z = f(A)
        }

        z
    }

    /// One training step on a minibatch: `xb` is (n, 784), `yb` is (n, 10).
    /// Returns the average loss of the batch, measured BEFORE the update.
    ///
    /// ```text
    /// forward:   A[l] = Z[l-1] · W[l] + b[l],  Z[l] = f(A[l])      (Z[-1] is xb)
    /// error:     E[L-1] = (P - Y) / n                              prediction minus answer
    /// backward:  E[l-1] = (E[l] · W[l]^T) * f'(A[l-1])             (* is element-wise)
    /// slopes:    dW[l] = Z[l-1]^T · E[l],  db[l] = sum of E[l]'s rows
    /// update:    W[l] -= lr * dW[l],  b[l] -= lr * db[l]
    /// ```
    pub fn train_step(&mut self, xb: &Matrix, yb: &Matrix) -> f64 {
        // ---- forward -------------------------------------------------------
        // Same as `predict`, but every layer's input Z[l-1] and pre-activation
        // A[l] are kept, because the backward pass needs them.
        let mut inputs: Vec<Matrix> = vec![]; // inputs[l] = Z[l-1], the input of layer l
        let mut pre_activations: Vec<Matrix> = vec![]; // pre_activations[l] = A[l]
        let mut z = xb.clone();

        for layer in &self.layers {
            let a = &z.dot(&layer.weights).unwrap() + &layer.bias;
            let next = (layer.activation)(&a);
            inputs.push(z);
            pre_activations.push(a);
            z = next;
        }
        let p = z; // the predicted probabilities, (n, 10)

        // ---- compare with the answer ---------------------------------------
        let loss = (self.loss_fn)(&p, yb);

        // Predicted minus actual is the error at the output: positive for every digit
        // the network rated too high, negative for the true digit if it was rated too low.
        // (This simple form holds because softmax is paired with cross-entropy loss:
        // their derivatives cancel out to exactly `P - Y`.)
        // Dividing by n makes the slopes below the AVERAGE over the batch, not the sum.
        let n = xb.rows as f64;
        let mut e = (&p - yb).scale(1.0 / n);

        // ---- backward and update, from the last layer to the first ----------
        for l in (0..self.layers.len()).rev() {
            // Each weight moves by its input times the error of its neuron.
            // `Z[l-1]^T · E` has exactly the shape of W, (inputs, neurons): the n images
            // meet in the middle and are summed away, one total per weight.
            // An input that was 0 (a blank pixel, or a feature that was off) adds
            // nothing to its weight's slope: "no ink, no blame".
            let d_weights = inputs[l].transposed().dot(&e).unwrap();
            // Each bias moves by the error of its neuron, added up over the batch.
            let d_bias = e.sum_columns();

            // Send the error back one layer, through the same weights turned round
            // (W^T), and through the derivative of the previous layer's activation.
            // This must use the weights BEFORE they are updated below.
            if l > 0 {
                let d_activation = self.layers[l - 1]
                    .d_activation
                    .expect("Hidden layers need a derivative of their activation");
                let back = e.dot(&self.layers[l].weights.transposed()).unwrap();
                e = &back * &d_activation(&pre_activations[l - 1]);
            }

            // The update: a small step against the slope.
            let layer = &mut self.layers[l];
            layer.weights = &layer.weights - &d_weights.scale(self.learning_rate);
            layer.bias = &layer.bias - &d_bias.scale(self.learning_rate);
        }

        loss
    }

    /// Saves the learned weights and biases as JSON: a list of `[weights, bias]`
    /// pairs, one per layer. (Activation functions are code, so they are not saved.)
    pub fn save(&self, path: &str) {
        let params: Vec<(&Matrix, &Matrix)> = self
            .layers
            .iter()
            .map(|layer| (&layer.weights, &layer.bias))
            .collect();

        let json = serde_json::to_string(&params).expect("Failed to serialise the model");
        std::fs::write(path, json).expect("Failed to write the model file");
    }
}
