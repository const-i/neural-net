//! A Neural Network library written completely in Rust implementing the back propagation
//! algorithm with stochastic gradient decent.
//!
//! # Description
//! This is a neural network library capable of learning through 
//! [Stochastic Gradient Decent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
//! using mini batches of the training data. This work is completely based on 
//! [this book](http://neuralnetworksanddeeplearning.com/) by [Micheal Nielsen](https://twitter.com/michael_nielsen).
//! Currently it only trains using the Sigmoid function with a quadratic cost
//! function, however that can be easily changed in the future. 
//!
//! # XOR Example
//!
//! This examples creates a neural network with 4 layers:
//!  - Input Layer with 2 nodes
//!  - Hidden Layer with 3 nodes
//!  - Hidden Layer with 3 nodes
//!  - Output Layer with 1 node
//! 
//! This network is then trained on the XOR function for 1000 epochs, using
//! all 4 inputs per training data in sets of 2, and a learning rate of 5.0
//!
//! ```
//! use neuralnet::NeuralNetworkBuilder;
//!
//! let sizes: Vec<usize> = vec![2, 3, 3, 1];
//! let inputs: Vec<Vec<f64>> = vec![
//!     vec![0.0, 0.0],
//!     vec![1.0, 0.0],
//!     vec![0.0, 1.0],
//!     vec![1.0, 1.0],
//! ];
//! let outputs: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
//!
//! let mut nn = NeuralNetworkBuilder::new(sizes, &inputs, &outputs)
//!                 .epochs(1000)
//!                 .mini_batch_size(2)
//!                 .learning_rate(5.0)
//!                 .train();
//!
//! for i in 0..inputs.len() {
//!     let nn_outputs = nn.feed_forward(&inputs[i]);
//!     assert!((nn_outputs[0] - outputs[i][0]).abs() < 0.1);
//! }
//! ```




use rand::seq::SliceRandom;
use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0_f64 - sigmoid(z))
}

fn cost_derivative(output_activation: f64, expected_value: f64) -> f64 {
    output_activation - expected_value
}

/// Used to store all the information for a single neuron in the network
#[derive(Debug)]
struct Neuron {
    value: f64,
    activation: f64,
    bias: f64,
    weights: Vec<f64>,
    del: f64,
    del_bias: f64,
    del_weight: Vec<f64>,
}

/// The base neuron which holds the its bias and weights for all the 
/// neurons in the layer before it.
impl Neuron {
    fn new(num_inputs: usize) -> Neuron {
        let mut rng = thread_rng();
        let distribution =
            Normal::new(0.0, 1.0).expect("Std-dev for Normal distribution should not be negative");
        //let distribution = Uniform::new(-1.0, 1.0);

        Neuron {
            value: 0.0,
            activation: 0.0,
            bias: distribution.sample(&mut rng),
            weights: distribution.sample_iter(rng).take(num_inputs).collect(),
            del: 0.0,
            del_bias: 0.0,
            del_weight: vec![0.0; num_inputs],
        }
    }
}


/// Used to store all the information for a single layer inside a neural network
struct Layer {
    num_neurons: usize,
    neurons: Vec<Neuron>,
}

/// The layer which stores the information of the neurons within it
impl Layer {
    fn new(num_inputs: usize, num_neurons: usize) -> Layer {
        Layer {
            num_neurons,
            neurons: (0..num_neurons).map(|_| Neuron::new(num_inputs)).collect(),
        }
    }
}

/// Used to store all the information for the neural network. It is built up of 
/// the activation function for the neurons (Sigmoid) and its derivative
/// (Sigmoid prime), the cost functions derivative (Quadratic Cost), the number
/// and sizes of the layers in the network and the actual layers (composed of neurons)
/// itself.
pub struct NeuralNetwork {
    activation_fn: fn(f64) -> f64,
    activation_derivative_fn: fn(f64) -> f64,
    cost_derivative_fn: fn(f64, f64) -> f64,
    num_layers: usize,
    sizes: Vec<usize>,
    layers: Vec<Layer>,
}

/// The Neural network is initialted through its builder (NeuralNetworkBuilder) and hence
/// all the parameters associated with it are present there.
impl NeuralNetwork {
    fn new(
        activation_fn: fn(f64) -> f64,
        activation_derivative_fn: fn(f64) -> f64,
        cost_derivative_fn: fn(f64, f64) -> f64,
        sizes: Vec<usize>,
    ) -> NeuralNetwork {
        assert!(!sizes.is_empty());
        assert!(sizes.iter().all(|&s| s > 0));

        let num_layers = sizes.len();
        let layers = (0..num_layers)
            .map(|i| {
                if i == 0 {
                    Layer::new(0, sizes[i])
                } else {
                    Layer::new(sizes[i - 1], sizes[i])
                }
            })
            .collect();
        NeuralNetwork {
            activation_fn,
            activation_derivative_fn,
            cost_derivative_fn,
            num_layers,
            sizes,
            layers,
        }
    }

    fn print(&self) {
        for i in 0..self.num_layers {
            for j in 0..self.layers[i].num_neurons {
                println!(
                    "Layer: {}; Neuron: {}; {:?}",
                    i, j, self.layers[i].neurons[j]
                );
            }
        }
    }

    fn propagate_inputs(&mut self, inputs: &[f64]) {
        assert_eq!(inputs.len(), self.layers[0].num_neurons);

        // Set values for the input layer
        for (n, &i) in self.layers[0].neurons.iter_mut().zip(inputs.iter()) {
            n.value = 0.0;
            n.activation = i;
        }

        // Feed Forward to the output layer
        for i in 1..self.num_layers {
            for j in 0..self.layers[i].num_neurons {
                let mut sum: f64 = 0.0;
                for k in 0..self.layers[i - 1].num_neurons {
                    sum += self.layers[i].neurons[j].weights[k]
                        * self.layers[i - 1].neurons[k].activation
                }
                sum += self.layers[i].neurons[j].bias;
                self.layers[i].neurons[j].value = sum;
                self.layers[i].neurons[j].activation = (self.activation_fn)(sum);
            }
        }
    }
	
	/// This function is used to get the output of the network for a specified set of inputs)
    pub fn feed_forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        // Propagate inputs
        self.propagate_inputs(inputs);

        // Return the outputs
        (0..self.layers[self.num_layers - 1].num_neurons)
            .map(|i| self.layers[self.num_layers - 1].neurons[i].activation)
            .collect()
    }

    fn back_propagate(&mut self, inputs: &[f64], outputs: &[f64]) {
        assert!(self.num_layers >= 2);
        assert_eq!(inputs.len(), self.layers[0].num_neurons);
        assert_eq!(outputs.len(), self.layers[self.num_layers - 1].num_neurons);

        // Propagate inputs
        self.propagate_inputs(inputs);

        // Calculate output layer errors
        for i in 0..self.layers[self.num_layers - 1].num_neurons {
            let del: f64 = (self.cost_derivative_fn)(
                self.layers[self.num_layers - 1].neurons[i].activation,
                outputs[i],
            ) * (self.activation_derivative_fn)(
                self.layers[self.num_layers - 1].neurons[i].value,
            );
            self.layers[self.num_layers - 1].neurons[i].del = del;
            self.layers[self.num_layers - 1].neurons[i].del_bias = del;
            for j in 0..self.layers[self.num_layers - 2].num_neurons {
                self.layers[self.num_layers - 1].neurons[i].del_weight[j] =
                    del * self.layers[self.num_layers - 2].neurons[j].activation;
            }
        }

        // Calculate errors for the rest of the layers
        for i in (1..(self.num_layers - 1)).rev() {
            for j in 0..self.layers[i].num_neurons {
                let mut del: f64 = 0.0;
                for k in 0..self.layers[i + 1].num_neurons {
                    del += self.layers[i + 1].neurons[k].del
                        * self.layers[i + 1].neurons[k].weights[j];
                }
                del *= (self.activation_derivative_fn)(self.layers[i].neurons[j].value); //value
                self.layers[i].neurons[j].del = del;
                self.layers[i].neurons[j].del_bias = del;
                for k in 0..self.layers[i - 1].num_neurons {
                    self.layers[i].neurons[j].del_weight[k] =
                        del * self.layers[i - 1].neurons[k].activation;
                }
            }
        }
    }

    fn update_mini_batch(&mut self, learning_rate: f64, samples: &[(&Vec<f64>, &Vec<f64>)]) {
        assert!(!samples.is_empty());

        // Collect the initial weights & biases
        let mut biases: Vec<Vec<f64>> = self
            .layers
            .iter()
            .map(|l| l.neurons.iter().map(|n| n.bias).collect())
            .collect();
        let mut weights: Vec<Vec<Vec<f64>>> = self
            .layers
            .iter()
            .map(|l| l.neurons.iter().map(|n| n.weights.clone()).collect())
            .collect();

        // Update the weights & biases based on the learning from this set of inputs
        let learning_factor: f64 = learning_rate / (samples.len() as f64);
        for (input, output) in samples {
            self.back_propagate(input, output);
            for i in 1..self.num_layers {
                for j in 0..self.layers[i].num_neurons {
                    biases[i][j] -= learning_factor * self.layers[i].neurons[j].del_bias;
                    for k in 0..self.layers[i - 1].num_neurons {
                        weights[i][j][k] -=
                            learning_factor * self.layers[i].neurons[j].del_weight[k];
                    }
                }
            }
        }

        // Update the weights & biases of the Neural network
        for i in 1..self.num_layers {
            for j in 0..self.layers[i].num_neurons {
                self.layers[i].neurons[j].bias = biases[i][j];
                for k in 0..self.layers[i - 1].num_neurons {
                    self.layers[i].neurons[j].weights[k] = weights[i][j][k];
                }
            }
        }
    }

    fn stochastic_gradient_decent(
        &mut self,
        inputs: &[Vec<f64>],
        outputs: &[Vec<f64>],
        epochs: usize,
        mini_batch_size: usize,
        learning_rate: f64,
    ) {
        assert!(!inputs.is_empty());
        assert_eq!(inputs.len(), outputs.len());
        assert!(mini_batch_size <= inputs.len());

        let mut samples: Vec<(&Vec<f64>, &Vec<f64>)> = inputs.iter().zip(outputs.iter()).collect();

        for _ in 0..epochs {
            samples.shuffle(&mut thread_rng());
            for mini_batch in samples.chunks(mini_batch_size) {
                self.update_mini_batch(learning_rate, mini_batch)
            }
        }
    }
}

/// Used to store all the information required to build and train a neural network. 
/// This includes the list of sizes of the layers, the inputs and outputs of a training
/// dataset, the number of epochs to run the training for, the size of the mini batch
/// of inputs to use at a time for the Stochastic Gradient Decent algorithm and 
/// learning rate (eta).
#[derive(Debug)]
pub struct NeuralNetworkBuilder<'a> {
	sizes: Vec<usize>,
	inputs: &'a [Vec<f64>],
	outputs: &'a [Vec<f64>],
	epochs: usize,
	mini_batch_size: usize, 
	learning_rate: f64,
}

/// The builder is responsible for building and training a neural network
impl<'a> NeuralNetworkBuilder<'a> {
	
	/// This function creates a new builder and takes the sizes of the layers
	/// and the inputs and outputs to train with.
	pub fn new(sizes: Vec<usize>, inputs: &'a [Vec<f64>], outputs: &'a [Vec<f64>]) -> NeuralNetworkBuilder<'a> {
		NeuralNetworkBuilder {
			epochs: 1000,
			mini_batch_size: inputs.len(), 
			learning_rate: 1.0,
			sizes,
			inputs,
			outputs,
		}
	}
	
	/// This specifies the epochs to train for. For every epoch the neural network will
	/// training using the entire training dataset once. 
	pub fn epochs(&mut self, epochs: usize) -> &mut NeuralNetworkBuilder<'a> {
		self.epochs = epochs;
		self
	}
	
	/// This sets the size of the batches of training data to use for every iteration of updating the 
	/// weights and biases of the network. This must be smaller than or equal to the length 
	/// of the training dataset.
	pub fn mini_batch_size(&mut self, mini_batch_size: usize) -> &mut NeuralNetworkBuilder<'a> {
		self.mini_batch_size = mini_batch_size;
		self
	}
	
	/// This sets the speed at which the neural network learns. Higher means that the network will
	/// learn faster but will be more prone to not converge, lower means that the network 
	/// will learn slower and need more number of epochs to converge but with higher
	/// probablity of converging.
	pub fn learning_rate(&mut self, learning_rate: f64) -> &mut NeuralNetworkBuilder<'a> {
		self.learning_rate = learning_rate;
		self
	}
	
	/// This is the terminal command to the builder and is used to create and train the neural
	/// network. It will return the fully trained neural network.
	pub fn train(&mut self) -> NeuralNetwork {
		let mut nn = NeuralNetwork::new(sigmoid, sigmoid_prime, cost_derivative, self.sizes.clone());
		nn.stochastic_gradient_decent(&self.inputs, &self.outputs, self.epochs, self.mini_batch_size, self.learning_rate);
		nn
	}
	
}




#[cfg(test)]
mod tests {

	use super::*;
	
    #[test]
    fn test_layer_stochastic_gradient_decent() {
        let sizes: Vec<usize> = vec![2, 3, 1];
        let mut nn = NeuralNetwork::new(sigmoid, sigmoid_prime, cost_derivative, sizes);
        let mut inputs: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let mut outputs: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
        let epochs: usize = 1000;
        let mini_batch_size: usize = 2;
        let learning_rate: f64 = 10.0;
        nn.stochastic_gradient_decent(
            &mut inputs,
            &mut outputs,
            epochs,
            mini_batch_size,
            learning_rate,
        );

        for i in 0..inputs.len() {
            let nn_outputs = nn.feed_forward(&inputs[i]);
            assert!((nn_outputs[0] - outputs[i][0]).abs() < 0.1);
            //println!("{:?}: {:.2}, ({:.2})", inputs[i], nn_outputs[0], outputs[i][0]);
        }
        //assert!(false);
    }
    
    #[test]
    fn test_network_builder() {
		
        let sizes: Vec<usize> = vec![2, 3, 3, 1];
        let inputs: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let outputs: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
		
		let mut nn = NeuralNetworkBuilder::new(sizes, &inputs, &outputs)
					.epochs(1000)
					.mini_batch_size(2)
					.learning_rate(5.0)
					.train();
		
		for i in 0..inputs.len() {
            let nn_outputs = nn.feed_forward(&inputs[i]);
            assert!((nn_outputs[0] - outputs[i][0]).abs() < 0.1);
            //println!("{:?}: {:.2}, ({:.2})", inputs[i], nn_outputs[0], outputs[i][0]);
        }
		//assert!(false);
	}
}
