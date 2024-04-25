use crate::matrix::{Matrix, Vector};

trait ActivationFunction {
    fn apply(&self, vector: &Vector) -> Vector;
    fn apply_derivative(&self, vector: &Vector) -> Vector;
}

struct Relu;

impl ActivationFunction for Relu {
    fn apply(&self, vector: &Vector) -> Vector {
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

struct FinalLayerInfo {
    y_actual: Vector,
}

struct EarlierLayerInfo {
    weights: Matrix,
    delta: Vector,
}

enum LayerInfo {
    Earlier(EarlierLayerInfo),
    Final(FinalLayerInfo),
}

#[derive(Clone)]
struct Layer {
    weights: Matrix,
    bias: Vector,
}

struct Gradients {
    weights: Matrix,
    bias: Vector,
    delta: Vector,
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
        let a = Relu.apply(&z);
        a
    }

    fn calculate_gradients(
        &self,
        inputs: &Vector,
        z: &Vector,
        layer_info: &LayerInfo,
    ) -> Gradients {
        let activation_derivatives = Relu.apply_derivative(z);
        let delta = match layer_info {
            LayerInfo::Final(FinalLayerInfo { y_actual }) => {
                let a = Relu.apply(z);
                mse_derivative(y_actual, &a).elementwise_mul(&activation_derivatives)
            }
            LayerInfo::Earlier(EarlierLayerInfo { weights, delta }) => {
                (weights.transpose() * delta).elementwise_mul(&activation_derivatives)
            }
        };
        let weights_gradient = delta.as_matrix() * inputs.as_matrix().transpose();
        let bias_gradient = delta.clone();
        Gradients {
            weights: weights_gradient,
            bias: bias_gradient,
            delta,
        }
    }

    pub fn backward(
        &mut self,
        inputs: &Vector,
        z: &Vector,
        layer_info: &LayerInfo,
        learning_rate: f32,
    ) -> EarlierLayerInfo {
        let original_weights = self.weights.clone();
        let Gradients {
            weights: weights_gradient,
            bias: bias_gradient,
            delta
        } = self.calculate_gradients(inputs, z, layer_info);
        self.weights -= weights_gradient * learning_rate;
        self.bias -= bias_gradient * learning_rate;

        EarlierLayerInfo {
            weights: original_weights,
            delta,
        }
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

    pub fn backward(&mut self, network_input: &Vector, y_actual: &Vector, learning_rate: f32) {
        let (a_vec, z_vec) = {
            let mut a_vec: Vec<Vector> = vec![];
            let mut z_vec: Vec<Vector> = vec![];
            for layer in self.layers.iter() {
                let input = {
                    match a_vec.last() {
                        Some(a) => a,
                        None => network_input,
                    }
                };
                let z = &layer.weights * input + &layer.bias;
                let a = Relu.apply(&z);
                z_vec.push(z);
                a_vec.push(a);
            }
            (a_vec, z_vec)
        };

        let mut layer_info = LayerInfo::Final(FinalLayerInfo { y_actual: y_actual.clone() });
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let inputs = if i == 0 { &network_input } else { &a_vec[i - 1] };
            let z = {
                z_vec[i].clone()
            };
            layer_info = LayerInfo::Earlier(layer.backward(
                inputs,
                &z,
                &layer_info,
                learning_rate,
            ));
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
        let learning_rate = 0.5;

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
        nn.backward(&inputs, &y_actual, learning_rate);

        // The new parameters are calculated by hand
        assert_eq!(nn.layers[1].weights, Matrix::from_data(vec![2.0, 2.0, 2.0, 4.0], 2, 2));
        assert_eq!(nn.layers[1].bias, Vector::new(vec![-0.5, -1.5]));

        assert_eq!(nn.layers[0].weights, Matrix::from_data(vec![0.0, 2.0, 2.0, -4.0, -5.0, -6.0], 2, 3));
        assert_eq!(nn.layers[0].bias, Vector::new(vec![-3.0, -2.0]));

        // Test if the NN works at all
        {
            let learning_rate = 0.001;
            let mut nn = make_nn();
            for _ in 0..10000 {
                nn.backward(&inputs, &y_actual, learning_rate);
            }
            let final_forward = nn.forward(&inputs);
            println!("Final forward: {:?}", final_forward);
            println!("Weights L0: {:?}", nn.layers[0].weights);
            println!("Weights L1: {:?}", nn.layers[1].weights);

            let loss = mse(&y_actual, &final_forward);
            assert!(loss < 0.0001, "Loss is too high: {}", loss)
        }
    }
}
