use crate::math::matrix::Matrix;

struct Layer {
    weights: Matrix,
    bias: Matrix,
    activation: fn(&Matrix) -> Matrix,
    d_activation: Option<fn(&Matrix) -> Matrix>,
}
struct FeedForwardStates {
    pre_activation: Vec<Matrix>,
    activations: Vec<Matrix>,
}

struct Gradients {
    errors: Vec<Matrix>,
    gradients: Vec<Matrix>,
}

pub struct Network {
    layers: Vec<Layer>,
    learning_rate: f64,
    loss_fn: fn(&Matrix, &Matrix) -> f64,
    feed_forward_states: FeedForwardStates,
    back_prop_states: Gradients,
}

impl Gradients {
    pub fn new() -> Self {
        Gradients {
            errors: vec![],
            gradients: vec![],
        }
    }
}

impl FeedForwardStates {
    pub fn new() -> Self {
        FeedForwardStates {
            pre_activation: vec![],
            activations: vec![],
        }
    }

    pub fn is_initialised(&self) -> bool {
        !(self.pre_activation.is_empty() && self.activations.is_empty())
    }
}

impl Network {
    pub fn new(learning_rate: f64, loss_fn: fn(&Matrix, &Matrix) -> f64) -> Network {
        Network {
            learning_rate,
            loss_fn,
            layers: vec![],
            feed_forward_states: FeedForwardStates::new(),
            back_prop_states: Gradients::new(),
        }
    }

    pub fn add_inp_layer(
        &mut self,
        neurons: usize,
        input_size: usize,
        activation: fn(&Matrix) -> Matrix,
        d_activation: Option<fn(&Matrix) -> Matrix>,
    ) {
        self.layers.push(Layer {
            weights: Matrix::uniform(neurons, input_size),
            activation,
            bias: Matrix::repeat(neurons, 1, 0.0),
            d_activation,
        });

        self.back_prop_states
            .gradients
            .push(Matrix::repeat(neurons, input_size, 0.0));

        self.back_prop_states
            .errors
            .push(Matrix::repeat(neurons, 1, 0.0));
    }

    pub fn add_layer(
        &mut self,
        neurons: usize,
        activation: fn(&Matrix) -> Matrix,
        d_activation: Option<fn(&Matrix) -> Matrix>,
    ) {
        if self.layers.is_empty() {
            panic!("Add an input layer before adding hidden layers");
        }

        let prev_rows = self.layers.last().unwrap().weights.rows();
        self.layers.push(Layer {
            weights: Matrix::uniform(neurons, prev_rows),
            activation,
            bias: Matrix::repeat(neurons, 1, 0.0),
            d_activation,
        });

        self.back_prop_states
            .gradients
            .push(Matrix::repeat(neurons, prev_rows, 0.0));

        self.back_prop_states
            .errors
            .push(Matrix::repeat(neurons, 1, 0.0));
    }

    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    pub fn feed_forward(&mut self, x: &Matrix, y: &Matrix) -> f64 {
        let f_s = &mut self.feed_forward_states;

        // Input layer
        let input_layer = self.layers.first().unwrap();
        let a = &input_layer.weights.dot(x).unwrap() + &input_layer.bias;
        let z = (input_layer.activation)(&a);
        f_s.pre_activation.push(a);
        f_s.activations.push(z);

        for i in 1..self.layers.len() {
            let hidden_layer = &self.layers[i];
            let a = &hidden_layer
                .weights
                .dot(f_s.activations.last().unwrap())
                .unwrap()
                + &hidden_layer.bias;
            let z = (hidden_layer.activation)(&a);
            f_s.pre_activation.push(a);
            f_s.activations.push(z);
        }

        (self.loss_fn)(f_s.activations.last().unwrap(), y)
    }

    pub fn calc_gradients(&mut self, x: &mut Matrix, y: &Matrix) {
        if !self.feed_forward_states.is_initialised() {
            panic!("Feed forward state not initialised, feed training data first.");
        }

        let f_s = &mut self.feed_forward_states;
        //todo: Calculate error using derivative of output activation and loss function
        let e = &f_s.activations.pop().unwrap() - y;
        let _ = f_s.pre_activation.pop().unwrap(); // Last layer's pre-activation is not required
        let mut z_prev = f_s.activations.pop().unwrap();
        z_prev.transpose();
        let grad = e.dot(&z_prev).unwrap();

        let g_s = &mut self.back_prop_states;
        let mut errors: Vec<Matrix> = vec![];

        errors.push(e);
        let depth = g_s.gradients.len();
        g_s.gradients[depth - 1] = g_s.gradients.last().unwrap() + &grad;

        for i in (1..self.layers.len() - 1).rev() {
            let d_activation = self.layers[i].d_activation;
            let next_layer = &mut self.layers[i + 1];
            next_layer.weights.transpose();

            let e_next_layer = errors.last().unwrap();
            let a = f_s.pre_activation.pop().unwrap();
            let e = &next_layer.weights.dot(&e_next_layer).unwrap() * &(d_activation.unwrap())(&a);
            let mut z_prev = f_s.activations.pop().unwrap();
            z_prev.transpose();
            let grad = e.dot(&z_prev).unwrap();
            g_s.gradients[i] = &g_s.gradients[i] + &grad;
            errors.push(e);

            next_layer.weights.transpose();
        }

        let inp_activation = &mut self.layers[0].d_activation.unwrap();
        let next_layer = &mut self.layers[1];
        next_layer.weights.transpose();
        let e_next_layer = errors.last().unwrap();
        let a = f_s.pre_activation.pop().unwrap();
        let e = &next_layer.weights.dot(&e_next_layer).unwrap() * &inp_activation(&a);
        x.transpose();
        let grad = e.dot(&x).unwrap();
        x.transpose();
        g_s.gradients[0] = &g_s.gradients[0] + &grad;
        errors.push(e);
        next_layer.weights.transpose();

        errors.reverse();

        for i in 0..g_s.errors.len() {
            g_s.errors[i] = &g_s.errors[i] + &errors[i];
        }
    }

    pub fn update_gradients(&mut self, batch_size: usize) {
        let g_s = &mut self.back_prop_states;

        for i in 0..g_s.gradients.len() {
            let layer = &mut self.layers[i];
            layer.weights = &layer.weights
                - &(&Matrix::repeat(
                    layer.weights.rows(),
                    layer.weights.cols(),
                    self.learning_rate / batch_size as f64,
                ) * &g_s.gradients[i]);

            layer.bias = &layer.bias
                - &(&Matrix::repeat(
                    layer.bias.rows(),
                    layer.bias.cols(),
                    self.learning_rate / batch_size as f64,
                ) * &g_s.errors[i]);
        }
    }
}
