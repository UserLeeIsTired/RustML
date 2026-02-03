use super::matrix::Matrix;

pub fn relu_derivative_helper(x: f32) -> f32 {
    if x <= 0.0 { 0.1 } else { 1.0 }
}

pub fn relu(input: &mut Matrix) {
    for val in input.arena.iter_mut() {
        if *val < 0.0 { *val *= 0.1; }
    }
}

pub fn softmax(input: &mut Matrix) {
    let max_val = input.arena.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    let mut sum = 1e-10;
    for i in 0..input.arena.len() {
        input.arena[i] = (input.arena[i] - max_val).exp();
        sum += input.arena[i];
    }

    for i in 0..input.arena.len() {
        input.arena[i] /= sum;
    }
}

pub fn cross_entropy_softmax_differentiate(output: &Matrix, correct_class: usize) -> Matrix {
    let (rows, cols) = output.shape();
    let mut grad = Matrix::new(rows, cols, false);
    
    for i in 0..output.arena.len() {
        grad.arena[i] = output.arena[i];
    }

    let target_idx = if rows == 1 {
        output.get_idx(0, correct_class)
    } else {
        output.get_idx(correct_class, 0)
    };

    grad.arena[target_idx] -= 1.0;
    grad
}