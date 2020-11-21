# neural-net

A Neural Network library written completely in Rust implementing the back propagation
algorithm with stochastic gradient decent.

### Description
This is a neural network library capable of learning through 
[Stochastic Gradient Decent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
using mini batches of the training data. This work is completely based on 
[this book](http://neuralnetworksanddeeplearning.com/) by [Micheal Nielsen](https://twitter.com/michael_nielsen).
Currently it only trains using the Sigmoid function with a quadratic cost
function, however that can be easily changed in the future. 

### XOR Example

This examples creates a neural network with 4 layers:
 - Input Layer with 2 nodes
 - Hidden Layer with 3 nodes
 - Hidden Layer with 3 nodes
 - Output Layer with 1 node

This network is then trained on the XOR function for 1000 epochs, using
all 4 inputs per training data in sets of 2, and a learning rate of 5.0

```
use neuralnet::NeuralNetworkBuilder;

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
}
```
