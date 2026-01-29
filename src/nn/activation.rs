use super::matrix::Matrix;

pub fn relu_derivative_helper(x: f32) -> f32 {
    if x <= 0.0 { 0.0 } else { 1.0 }
}

pub fn relu(input: &mut Matrix) {
    for val in input.arena.iter_mut() {
        if *val < 0.0 { *val = 0.0; }
    }
}

pub fn softmax(input: &mut Matrix) {
    let max_val = input.arena.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.arena.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for (val, exp) in input.arena.iter_mut().zip(exps.iter()) {
        *val = exp / sum;
    }
}

pub fn cross_entropy_softmax_differentiate(output: &Matrix, correct_class: usize) -> Matrix {
    let mut grad = Matrix::new(1, output.col, false);
    for i in 0..output.col {
        grad.arena[i] = output.arena[i];
        if i == correct_class {
            grad.arena[i] -= 1.0;
        }
    }
    grad
}