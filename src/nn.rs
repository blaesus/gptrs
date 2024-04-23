use crate::matrix::{Matrix, Vector};

fn relu_f32(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn relu_vector(vector: Vector) -> Vector {
    let data = vector.data().iter().map(|x| relu_f32(*x)).collect();
    Vector::new(data)
}

#[derive(Clone)]
struct Layer {
    weights: Matrix,
    bias: Vector,
}

impl Layer {
    pub fn new_kaiming(input_size: usize, output_size: usize) -> Layer {
        Layer {
            weights: Matrix::new_kaiming(output_size, input_size),
            bias: Vector::new_kaiming(output_size),
        }
    }

    pub fn normalize(&mut self) {
        self.bias.normalize();
    }
}

#[derive(Clone)]
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
            let Layer {bias, weights} = layer;
            result = relu_vector(weights * result + bias);
        }
        result
    }

    pub fn backward(&self, inputs: Vector, targets: Vector) {
        let result = self.forward(&inputs);
        let error = targets - result;
        let mut delta = error.clone();
        // for layer in self.layers.iter().rev() {
        //     let weights_transposed = layer.weights.clone().transpose();
        //     let gradient = delta.clone() * result.clone();
        //     let delta = weights_transposed * delta;
        // }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nn_feedforward() {
        let layer1 = Layer {
            weights: Matrix::from_data(vec![1.0, 2.0, 3.0, -4.0], 2, 2),
            bias: Vector::new_uniform(1.0, 2),
        };
        let layer2 = Layer {
            weights: Matrix::from_data(vec![1.0, 2.0], 1, 2),
            bias: Vector::new_uniform(1.0, 1),
        };
        let nn = NeuralNetwork::new(vec![layer1, layer2]);
        let inputs = Vector::new(vec![1.0, 2.0]);
        let result = nn.forward(&inputs);
        assert_eq!(result.data(), &vec![7.0]);
    }
}
