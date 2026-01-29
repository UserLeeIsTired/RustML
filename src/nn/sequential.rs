use crate::nn::matrix::Matrix;

use super::layer::Layer;
use super::activation::cross_entropy_softmax_differentiate;

pub struct Sequential {
    learning_rate: f32,
    pub layers: Vec<Layer>
}

impl Sequential {
    pub fn new(learning_rate: f32, layers: Vec<Layer>) -> Self {
        Sequential { 
            learning_rate: learning_rate,
            layers: layers,
        }
    }

    pub fn forward(&mut self, mut input: Matrix) -> Matrix {
        
        for layer in &mut self.layers {
            input = layer.forward(input);
        }

        input
    }

    pub fn backward(&mut self, output: Matrix, true_label: usize) {
        let mut error_signal = cross_entropy_softmax_differentiate(
            &output, 
            true_label
        );

    for layer in self.layers.iter_mut().rev() {
        error_signal = layer.backward(error_signal, self.learning_rate);
    }
}
}