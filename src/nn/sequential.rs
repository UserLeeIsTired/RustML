use crate::nn::data::{TestSet, TrainSet, test_set, train_set};
use crate::nn::matrix::Matrix;

use super::layer::Layer;
use super::activation::cross_entropy_softmax_differentiate;

pub struct Sequential {
    learning_rate: f32,
    batch_size: usize,
    pub layers: Vec<Layer>
}

impl Sequential {
    pub fn new(learning_rate: f32, batch_size: usize, layers: Vec<Layer>) -> Self {
        Sequential { 
            learning_rate: learning_rate,
            batch_size: batch_size,
            layers: layers,
        }
    }

    pub fn forward(&mut self, mut input: Matrix) -> Matrix {
        
        for layer in &mut self.layers {
            input = layer.forward(input);
        }

        input
    }

    pub fn update(&mut self) {
        for layer in &mut self.layers {
            layer.update();
        }
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

    pub fn train(&mut self, epoch: usize, mut train_set: TrainSet, test_set: TestSet, verbose: bool) {
        for i in 0..epoch {
            while train_set.signal() {
                for input in train_set.get() {
                    let result = self.forward(Matrix::to_matrix(input, 1));
                    self.backward(result, input[0] as usize);
                }
                self.update();
                if verbose {println!("[{}/{}][{}/{}] completed", i+1,epoch, train_set.pointer, train_set.n)}
            }


            let mut correct = 0;
            for input in test_set.get() {
                let result = self.forward(Matrix::to_matrix(input, 1));
                let (_, y) = result.get_max();
                if y == input[0] as usize { correct += 1; }
            }
            if verbose {println!("Epoch {} completed - {}% accuracy", epoch+1, correct as f32 / train_set.n as f32 * 100.0)}

        }
    }
}