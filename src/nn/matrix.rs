use std::f32::consts::PI;
use rand::prelude::*;

pub struct Matrix {
    pub row: usize,     // Physical Rows
    pub col: usize,     // Physical Columns
    pub arena: Vec<f32>,
    pub transpose: bool,
}

fn box_muller_f32() -> f32 {
    let mut rng = rand::rng(); 
    let u1: f32 = 1.0 - rng.random::<f32>(); 
    let u2: f32 = rng.random::<f32>();
    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;
    radius * theta.cos()
}

impl Matrix {
    pub fn new(row: usize, col: usize, req_init: bool) -> Self {
        let size = row * col;
        let arena = (0..size)
            .map(|_| if req_init { box_muller_f32() * (2.0 / row as f32).sqrt() } else { 0.0 })
            .collect();
        
        Self { row, col, arena, transpose: false }
    }

    /// Returns the logical dimensions (respecting the transpose flag)
    pub fn shape(&self) -> (usize, usize) {
        if self.transpose { (self.col, self.row) } else { (self.row, self.col) }
    }

    // Maps logical (r, c) to the physical index in the arena
    pub fn get_idx(&self, r: usize, c: usize) -> usize {
        let idx = if self.transpose {
            c * self.col + r
        } else {
            r * self.col + c
        };
        
        idx
    }

    pub fn to_matrix(vector: &[u8], skip: usize) -> Self {
        Self {
            row: 1,
            col: vector.len() - 1,
            arena: vector.iter().skip(skip).map(|&x| x as f32 / 255.0).collect(),
            transpose: false,
        }
    }

    pub fn transpose(&mut self) {
        self.transpose = !self.transpose;
    }

    pub fn dot(&self, rhs: &Matrix) -> Matrix {
        let (r1, c1) = self.shape();
        let (r2, c2) = rhs.shape();
        
        if c1 != r2 {
            panic!("Dim mismatch: {}x{} * {}x{}", r1, c1, r2, c2); 
        }
        
        let mut result = Matrix::new(r1, c2, false);
        
        for i in 0..r1 {
            for j in 0..c2 {
                let mut sum = 0.0;
                for k in 0..c1 {
                    sum += self.arena[self.get_idx(i, k)] * rhs.arena[rhs.get_idx(k, j)];
                }
                // Use get_idx even here to avoid manual math errors
                let target_idx = result.get_idx(i, j);
                result.arena[target_idx] = sum;
            }
        }
        result
    }

    pub fn add(&self, rhs: &Matrix) -> Matrix {
        let (r1, c1) = self.shape();
        let (r2, c2) = rhs.shape();

        if (r1, c1) != (r2, c2) {
            panic!("Add mismatch: {}x{} + {}x{}", r1, c1, r2, c2);
        }
        
        let mut result = Matrix::new(r1, c1, false);
        for i in 0..r1 {
            for j in 0..c1 {
                let val = self.arena[self.get_idx(i, j)] + rhs.arena[rhs.get_idx(i, j)];
                // Use the result's own indexing logic to be safe
                let target_idx = result.get_idx(i, j); 
                result.arena[target_idx] = val;
            }
        }
        result
    }

    pub fn backward_relu(mut self, z_mask: &Matrix) -> Self {
        // Since this is element-wise, we can ignore shape as long as sizes match
        for (error, &z) in self.arena.iter_mut().zip(z_mask.arena.iter()) {
            if z <= 0.0 { *error *= 0.01; } // Leaky ReLU logic
        }
        self
    }

    pub fn get_max(&self) -> (usize, usize) {
        let (rows, cols) = self.shape();
        let mut best_coord = (0, 0);
        let mut max_val = f32::NEG_INFINITY;

        for r in 0..rows {
            for c in 0..cols {
                let val = self.arena[self.get_idx(r, c)];
                if val > max_val {
                    max_val = val;
                    best_coord = (r, c);
                }
            }
        }
        best_coord
    }
}

// Boilerplate for operator overloading
impl<'a, 'b> std::ops::Mul<&'b Matrix> for &'a Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &'b Matrix) -> Self::Output { self.dot(rhs) }
}

impl<'a, 'b> std::ops::Add<&'b Matrix> for &'a Matrix {
    type Output = Matrix;
    fn add(self, rhs: &'b Matrix) -> Self::Output { self.add(rhs) }
}