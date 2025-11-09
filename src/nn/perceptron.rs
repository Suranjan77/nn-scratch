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

    pub fn add_layer(&mut self, neurons: usize, activation: fn(&Matrix) -> Matrix, d_activation: Option<fn(&Matrix) -> Matrix>) {
        if self.layers.is_empty() {
            self.layers.push(Layer {
                weights: Matrix::uniform(neurons, 1),
                activation,
                bias: Matrix::repeat(neurons, 1, 1.0),
                d_activation,
            });
        } else {
            let prev = self.layers.last().unwrap();
            self.layers.push(Layer {
                weights: Matrix::uniform(neurons, prev.weights.rows()),
                activation,
                bias: Matrix::repeat(neurons, 1, 1.0),
                d_activation,
            })
        }
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

    // todo
    pub fn calc_gradients(&mut self, x: &Matrix, y: &Matrix) {
        if (!self.feed_forward_states.is_initialised()) {
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
        g_s.errors.push(e);
        g_s.gradients.push(grad);

        for i in (1..self.layers.len() - 1).rev() {
            let hidden_layer = &mut self.layers[i];
            hidden_layer.weights.transpose();

            let e_next_layer = g_s.errors.last().unwrap();
            let a = f_s.pre_activation.pop().unwrap();
            let e = match hidden_layer.d_activation {
                Some(ref d) => &hidden_layer.weights.dot(&e_next_layer).unwrap() * &d(&a),
                None => hidden_layer.weights.dot(&e_next_layer).unwrap()
            };
            let mut z_prev = f_s.activations.pop().unwrap();
            z_prev.transpose();
            let grad = e.dot(&z_prev).unwrap();
            g_s.gradients.push(grad);
            g_s.errors.push(e);

            hidden_layer.weights.transpose();
        }


    }
}
