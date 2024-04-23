use crate::matrix::{Matrix, Vector};

trait ActivationFunction {
    fn apply(&self, vector: Vector) -> Vector;
    fn apply_derivative(&self, vector: Vector) -> Vector;
}

struct Relu;

impl ActivationFunction for Relu {

    fn apply(&self, vector: Vector) -> Vector {
        let data = vector.data().iter().map(|x| x.max(0.0)).collect();
        Vector::new(data)
    }

    fn apply_derivative(&self, vector: Vector) -> Vector {
        let data = vector.data().iter().map(|x| if *x > 0.0 { 1.0 } else { 0.0 }).collect();
        Vector::new(data)
    }
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

    pub fn forward(&self, inputs: &Vector) -> Vector {
        let Layer {bias, weights} = self;
        let z = weights * inputs + bias;
        let a = Relu.apply(z);
        a
    }

    pub fn backward(&mut self) {
        let learning_rate = 0.001;
        let Layer {bias, weights} = self;

        let gradient_weight = todo!();
        let new_weights = weights - learning_rate * gradient_weight;
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
        for layer in self.layers.iter() {
            result = layer.forward(&result);
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
