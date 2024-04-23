use crate::matrix::Matrix;

struct Layer {
    weights: Matrix,
    bias: f32,
}

struct NeuralNetwork {
    layers: Vec<Layer>
}

