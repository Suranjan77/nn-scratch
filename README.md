# nn-scratch

A neural network written from scratch in Rust. It is a small multilayer
perceptron that learns to read handwritten digits from the
[MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset.

No ML framework is used. The matrix type, activation functions, loss functions,
backpropagation and the MNIST file parser are all in this repository. The only
outside crates are helpers: fast matrix multiplication (`matrixmultiply`),
random numbers (`rand`) and saving the model (`serde`).

A live visualisation of the network training (weights, biases and data moving
through it) is on the `visualisation` branch.

## Run it

```sh
cargo run --release
```

Use `--release`: a debug build is many times slower. On a laptop a release run
takes about 10 seconds and reaches about 96% test accuracy.

It prints the shapes of the data and of each layer, one line per epoch
(training loss, validation accuracy, time), and the test accuracy at the end.
The trained weights are saved to `model.json`.

## The network

```text
784 pixels ──► 256 sigmoid ──► 64 sigmoid ──► 10 softmax
              W1 (784, 256)   W2 (256, 64)   W3 (64, 10)      218,058 parameters
```

- **Input**: each 28×28 image becomes one row of 784 numbers, scaled to 0–1.
- **Hidden layers** turn pixels into features.
- **Output**: softmax turns 10 scores into 10 probabilities, one per digit.
- **Loss**: cross-entropy.
- **Training**: minibatch gradient descent, 100 images per batch, learning
  rate 1.0, 5 epochs.

Everything works on whole batches using the design matrix (one row per image).
One training step is:

```text
forward:   A = Z·W + b,  Z = f(A)              layer by layer
error:     E = (P − Y) / n                     prediction minus answer
backward:  E_prev = (E · Wᵀ) ∗ f'(A_prev)
update:    W −= lr · Z_prevᵀ · E,   b −= lr · sum of E's rows
```

The data is split three ways. The **training** set (50,000 images) updates the
weights. The **validation** set (10,000) is watched during training to choose
settings. The **test** set (10,000) is scored once at the end, so its score is
an honest estimate of how the network does on digits it has never seen.

## Project layout

```text
src/
├── main.rs                 training run: load → build → train → report
├── nn/perceptron.rs        Network: layers, forward pass, backprop, save
├── math/matrix.rs          Matrix type (strided, free transpose, matmul)
├── math/activation.rs      sigmoid, relu, softmax and their derivatives
├── math/loss_functions.rs  cross-entropy, binary cross-entropy, SSE
└── data/idx_parser.rs      reader for MNIST's IDX file format
mnist_data/                 the MNIST training and test files
```

## Things to try

The hyperparameters are at the top of the "Build the network" and "Train"
sections in `src/main.rs`:

- change `epochs`, `batch_size` or the learning rate and watch the validation
  accuracy
- add, remove or resize layers with `add_layer`
- swap `sigmoid`/`d_sigmoid` for `relu`/`d_relu` in the hidden layers
