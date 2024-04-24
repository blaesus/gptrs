use crate::matrix::{Matrix, Vector};

trait ActivationFunction {
    fn apply(&self, vector: Vector) -> Vector;
    fn apply_derivative(&self, vector: &Vector) -> Vector;
}

struct Relu;

impl ActivationFunction for Relu {
    fn apply(&self, vector: Vector) -> Vector {
        let data = vector.data().iter().map(|x| x.max(0.0)).collect();
        Vector::new(data)
    }

    fn apply_derivative(&self, vector: &Vector) -> Vector {
        let data = vector.data().iter().map(|x| if *x > 0.0 { 1.0 } else { 0.0 }).collect();
        Vector::new(data)
    }
}

fn mse(y_actual: &Vector, y_predicted: &Vector) -> f32 {
    let sum: f32 = y_actual.data().iter().zip(y_predicted.data()).map(|(a, b)| (a - b).powi(2)).sum();
    sum / y_actual.data().len() as f32
}

fn mse_derivative(y_actual: &Vector, y_predicted: &Vector) -> Vector {
    let n = y_actual.data().len() as f32;
    let data = y_actual.data().iter().zip(y_predicted.data()).map(|(a, b)| 2.0 * (b - a) / n).collect();
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

    pub fn forward(&self, inputs: &Vector) -> Vector {
        let Layer { bias, weights } = self;
        let z = weights * inputs + bias;
        let a = Relu.apply(z);
        a
    }

    pub fn backward(
        &mut self,
        inputs: &Vector,
        y_calculated: &Vector,
        y_actual: &Vector,
        learning_rate: f32,
    ) -> Vector {
        // Magic! This is just the vectorized form of partial derivatives of loss relative to each
        // element in the upstream matrices/vectors.
        let delta = mse_derivative(y_actual, y_calculated).elementwise_mul(&Relu.apply_derivative(y_calculated));
        let weights_gradient = delta.as_matrix() * inputs.as_matrix().transpose();
        let bias_gradient = delta.clone();
        let previous_input_gradient = self.weights.transpose() * delta;

        self.weights = self.weights.clone() - weights_gradient * learning_rate;
        self.bias = self.bias.clone() - bias_gradient * learning_rate;

        previous_input_gradient
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

    pub fn backward(&mut self, inputs: &Vector, y_actual: &Vector, learning_rate: f32) {
        let layer_original_outputs = {
            let mut outputs: Vec<Vector> = vec![];
            for (i, layer) in self.layers.iter().enumerate() {
                let input = if i == 0 { inputs.clone() } else { outputs[i - 1].clone() };
                let y_calculated = layer.forward(&input);
                outputs.push(y_calculated.clone());
            }
            outputs
        };

        let layers_count = self.layers.len();
        let mut previous_input_gradient = Vector::new(vec![]);
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let inputs = {
                if i == 0 {
                    inputs.clone()
                } else {
                    layer_original_outputs[i - 1].clone()
                }
            };
            let y_calculated = {
                layer_original_outputs[i].clone()
            };
            let y_expected = {
                if i == layers_count - 1 {
                    y_actual.clone()
                } else {
                    let diff = previous_input_gradient * learning_rate;
                    &y_calculated - &diff
                }
            };
            previous_input_gradient = layer.backward(
                &inputs,
                &y_calculated,
                &y_expected,
                learning_rate,
            );
        }
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

    #[test]
    fn test_nn_backward_manual_case() {

        fn make_nn() -> NeuralNetwork {
            let layer1 = Layer {
                weights: Matrix::from_data(vec![1.0, 2.0, 3.0, -4.0, -5.0, -6.0], 2, 3),
                bias: Vector::new_uniform(-2.0, 2),
            };
            let layer2 = Layer {
                weights: Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2),
                bias: Vector::new_uniform(-1.0, 2),
            };
            NeuralNetwork::new(vec![layer1, layer2])
        }

        let mut nn = make_nn();

        let inputs = Vector::new(vec![1.0, 0.0, 1.0]);
        let forward_1 = nn.forward(&inputs);
        assert_eq!(forward_1.data(), &vec![1.0, 5.0]);
        let y_actual = Vector::new(vec![2.0, 4.0]);
        nn.backward(&inputs, &y_actual, 0.5);

        // The new parameters are calculated by hand
        assert_eq!(nn.layers[1].weights, Matrix::from_data(vec![2.0, 2.0, 2.0, 4.0], 2, 2));
        assert_eq!(nn.layers[1].bias, Vector::new(vec![-0.5, -1.5]));

        assert_eq!(nn.layers[0].weights, Matrix::from_data(vec![0.5, 2.0, 2.5, -4.0, -5.0, -6.0], 2, 3));
        assert_eq!(nn.layers[0].bias, Vector::new(vec![-2.5, -2.0]));

        // Test if the NN works at all
        {
            let mut nn = make_nn();
            for _ in 0..1000 {
                nn.backward(&inputs, &y_actual, 0.001);
            }
            let final_forward = nn.forward(&inputs);
            let loss = mse(&y_actual, &final_forward);
            assert!(loss < 0.0001, "Loss is too high: {}", loss)
        }

    }
}
