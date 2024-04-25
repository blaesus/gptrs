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
    let n = y_actual.size() as f32;
    let data = y_actual.data().iter().zip(y_predicted.data()).map(|(a, b)| 2.0 * (b - a) / n).collect();
    Vector::new(data)
}


#[derive(Debug, Clone)]
struct FinalLayerInfo {
    y_actual: Vector,
}

impl From<FinalLayerInfo> for LayerInfo {
    fn from(info: FinalLayerInfo) -> Self {
        LayerInfo::Final(info)
    }
}

impl From<EarlierLayerInfo> for LayerInfo {
    fn from(info: EarlierLayerInfo) -> Self {
        LayerInfo::Earlier(info)
    }
}

#[derive(Debug, Clone)]
struct EarlierLayerInfo {
    weights: Matrix,
    delta: Vector,
}

#[derive(Debug, Clone)]
enum LayerInfo {
    Earlier(EarlierLayerInfo),
    Final(FinalLayerInfo),
}

#[derive(Debug, Clone)]
struct Layer {
    weights: Matrix,
    bias: Vector,
}

#[derive(Debug, Clone)]
struct Gradients {
    weights: Matrix,
    bias: Vector,
    delta: Vector,
}

#[derive(Debug, Clone)]
struct DataPoint {
    input: Vector,
    output: Vector,
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
        self.weights.normalize();
    }

    pub fn forward(&self, inputs: &Vector) -> Vector {
        let Layer { bias, weights } = self;
        let z = weights * inputs + bias;
        let a = Relu.apply(&z);
        a
    }

    fn calculate_gradients(
        &self,
        input: &Vector,
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
        let weights_gradient = delta.as_matrix() * input.as_matrix().transpose();
        let bias_gradient = delta.clone();
        Gradients {
            weights: weights_gradient,
            bias: bias_gradient,
            delta,
        }
    }

    // SGD. Clean and simple.
    pub fn backward_single(
        &mut self,
        input: &Vector,
        z: &Vector,
        layer_info: &LayerInfo,
        learning_rate: f32,
    ) -> EarlierLayerInfo {
        let original_weights = self.weights.clone();
        let Gradients {
            weights: weights_gradient,
            bias: bias_gradient,
            delta
        } = self.calculate_gradients(input, z, layer_info);
        self.weights -= weights_gradient * learning_rate;
        self.bias -= bias_gradient * learning_rate;

        EarlierLayerInfo {
            weights: original_weights,
            delta,
        }
    }

    pub fn backward_batched(
        &mut self,
        data: &[(DataPoint, LayerInfo)],
        learning_rate: f32,
    ) -> Vec<EarlierLayerInfo> {
        let original_weights = self.weights.clone();
        let mut accumulated_weight_gradient = Matrix::zeros(self.weights.rows, self.weights.cols);
        let mut accumulated_bias_gradient = Vector::zeros(self.bias.size());
        let mut deltas = vec![];

        for point in data {
            let (DataPoint { input, output: z }, downstream) = point;
            let Gradients {
                weights: weights_gradient,
                bias: bias_gradient,
                delta: calculated_delta
            } = self.calculate_gradients(input, z, downstream);
            accumulated_weight_gradient += weights_gradient;
            accumulated_bias_gradient += bias_gradient;
            deltas.push(calculated_delta);
        }
        let averaged_weights_gradient = accumulated_weight_gradient / data.len() as f32;
        let averaged_bias_gradient = accumulated_bias_gradient / data.len() as f32;

        self.weights -= averaged_weights_gradient * learning_rate;
        self.bias -= averaged_bias_gradient * learning_rate;

        deltas
            .into_iter()
            .map(|delta| EarlierLayerInfo { weights: original_weights.clone(), delta })
            .collect()
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

        let mut downstream_layer = LayerInfo::Final(FinalLayerInfo { y_actual: y_actual.clone() });
        for (l, layer) in self.layers.iter_mut().enumerate().rev() {
            let layer_input = if l == 0 { &network_input } else { &a_vec[l - 1] };
            let layer_z = &z_vec[l];
            downstream_layer = LayerInfo::Earlier(layer.backward_single(
                layer_input,
                layer_z,
                &downstream_layer,
                learning_rate,
            ));
        }
    }

    pub fn backward_batched(&mut self, batch: &[DataPoint], learning_rate: f32) {
        // 1. Accumulate activations (a) and pre-activations (z) for all layers across the entire batch.
        let (a_batch_vec, z_batch_vec): (Vec<Vec<Vector>>, Vec<Vec<Vector>>) = {
            let mut a_batch_vec: Vec<Vec<Vector>> = Vec::new();
            let mut z_batch_vec: Vec<Vec<Vector>> = Vec::new();

            for data_point in batch {
                let mut a_layer_batch: Vec<Vector> = Vec::new();
                let mut z_layer_batch: Vec<Vector> = Vec::new();

                for layer in self.layers.iter() {
                    let input = {
                        match a_layer_batch.last() {
                            Some(a) => a,
                            None => &data_point.input,
                        }
                    };
                    let z = &layer.weights * input + &layer.bias;
                    let a = Relu.apply(&z);
                    z_layer_batch.push(z);
                    a_layer_batch.push(a);
                }
                z_batch_vec.push(z_layer_batch);
                a_batch_vec.push(a_layer_batch);
            }
            (a_batch_vec, z_batch_vec)
        };

        // 2. Perform backward pass for each layer, starting from the output back to the first layer.
        let mut downstream_layers: Vec<LayerInfo> = batch.iter().map(
            |point| FinalLayerInfo { y_actual: point.output.clone() }.into()
        ).collect();
        for (l, layer) in self.layers.iter_mut().enumerate().rev() {
            let inputs_and_zs: Vec<DataPoint> = batch.iter().enumerate().map(|(batch_index, _)| DataPoint {
                input: (if l == 0 { &batch[batch_index].input } else { &a_batch_vec[batch_index][l-1] }).clone(),
                output: z_batch_vec[batch_index][l].clone(),
            }).collect();

            let data: Vec<(DataPoint, LayerInfo)> = inputs_and_zs.into_iter()
                .zip(downstream_layers.iter())
                .map(|(data_point, downstream)| (data_point, downstream.clone()))
                .collect();

            // Call the batched version of the backward function for each layer.
            downstream_layers = layer.backward_batched(&data, learning_rate).into_iter().map(|info| info.into()).collect();
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::rand::{random_f32, random_gaussian};
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

    fn random_mini_batch(mut data: &[DataPoint], percentage: f32) -> Vec<DataPoint> {
        let n = (data.len() as f32 * percentage).round() as usize;
        let mut indices = (0..data.len()).collect::<Vec<_>>();
        for i in 0..n {
            let j = (random_f32() * data.len() as f32) as usize;
            indices.swap(i, j);
        }
        indices.truncate(n);
        indices.iter().map(|&i| data[i].clone()).collect()
    }

    #[test]
    fn test_nn_backward_batched() {
        let learning_rate = 0.5;

        fn make_nn() -> NeuralNetwork {
            let mut layer1 = Layer {
                weights: Matrix::new_random(5, 3),
                bias: Vector::new_random(5),
            };
            let mut layer3 = Layer {
                weights: Matrix::new_random(2, 5),
                bias: Vector::new_random(2),
            };
            layer1.normalize();
            layer3.normalize();
            NeuralNetwork::new(vec![layer1, layer3])
        }


        {
            let learning_rate = 0.001;
            let mut nn = make_nn();
            // x=a+b+1, y=b+c+2
            let data = {
                let mut data = vec![];
                for _ in 0..100000 {
                    let a = random_gaussian(0.0, 1.0);
                    let b = random_gaussian(1.0, 2.0);
                    let c = random_gaussian(-2.0, 1.0);
                    let x = a + b + 1.0;
                    let y = b + c + 2.0;
                    data.push(DataPoint {
                        input: Vector::new(vec![a, b, c]),
                        output: Vector::new(vec![x, y]),
                    });
                }
                data
            };

            let inputs = Vector::new(vec![3.0, 2.0, -1.0]);
            for _ in 0..1000 {
                let mini_batch = random_mini_batch(&data, 0.1);
                nn.backward_batched(&mini_batch, learning_rate);
                let forward = nn.forward(&inputs);
                println!("Forward {:?}", forward);
            }
            let final_forward = nn.forward(&inputs);
            let loss = mse(&Vector::new(vec![6.0, 3.0]), &final_forward);
            assert_eq!(loss, 0.01, "Loss is too high: {}", loss);
            println!("Final forward: {:?}", final_forward);
            println!("Weights L0: {:?}", nn.layers[0].weights);
        }
    }
}
