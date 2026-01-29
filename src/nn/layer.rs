use crate::nn::activation::{relu, softmax};
use super::matrix::Matrix;

pub struct Layer {
    pub weight: Matrix,
    pub bias: Matrix,
    pub last_input: Option<Matrix>,
    pub last_z: Option<Matrix>,
    activation_function: &'static str,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: &'static str) -> Self {
        Layer {
            weight: Matrix::new(input_size, output_size, true),
            bias: Matrix::new(1, output_size, false),
            last_input: None,
            last_z: None,
            activation_function: activation,
        }
    }

    pub fn forward(&mut self, input: Matrix) -> Matrix {
        let mut z = &(&input * &self.weight) + &self.bias;
        self.last_input = Some(input);
        self.last_z = Some(Matrix {
            row: z.row,
            col: z.col,
            arena: z.arena.clone(),
        });

        match self.activation_function {
            "relu" => relu(&mut z),
            "softmax" => softmax(&mut z),
            _ => panic!("Illegal activation")
        }
        z
    }

    pub fn backward(&mut self, feed_backward: Matrix, learning_rate: f32) -> Matrix {
        let mut last_input = self.last_input.take().expect("No input");
        let last_z = self.last_z.take().expect("No z");

        self.weight.transpose();
        
        let passback =
        match self.activation_function {
            "relu" => (&feed_backward * &self.weight).backward_relu(&last_z),
            "softmax" => (&feed_backward * &self.weight),
            _ => panic!("Illegal activation"),
        };
        
        self.weight.transpose();

        for i in 0..self.bias.arena.len() {
            self.bias.arena[i] -= learning_rate * feed_backward.arena[i];
        }

        last_input.transpose();
        let gradient = &last_input * &feed_backward;
        for i in 0..self.weight.arena.len() {
            self.weight.arena[i] -= learning_rate * gradient.arena[i];
        }

        passback
    }
}