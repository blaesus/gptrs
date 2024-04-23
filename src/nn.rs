use crate::matrix::{Matrix, Vector};

struct Layer {
    weights: Matrix,
    bias: f32,
}

struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> NeuralNetwork {
        NeuralNetwork { layers }
    }

    pub fn forward(&self, inputs: &Vector) -> Vector {
        let mut result = inputs.clone();
        for layer in &self.layers {
            let bias= Vector::new_uniform(layer.bias, layer.weights.rows);
            result = layer.weights.clone() * result + bias
        }
        result
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nn_feedforward() {
        let layer1 = Layer {
            weights: Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2),
            bias: 1.0,
        };
        let layer2 = Layer {
            weights: Matrix::from_data(vec![1.0, 2.0], 1, 2),
            bias: 1.0,
        };
        let nn = NeuralNetwork::new(vec![layer1, layer2]);
        let inputs = Vector::new(vec![1.0, 2.0]);
        let result = nn.forward(&inputs);
        assert_eq!(result.data(), &vec![31.0]);
    }
}
