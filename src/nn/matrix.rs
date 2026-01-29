use core::f32;
use std::ops::{Add, Mul};
use std::f32::consts::PI;
use rand::{prelude::*};
use super::activation::relu_derivative_helper;

pub struct Matrix {
    pub row: usize,
    pub col: usize,
    pub arena: Vec<f32>,
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
        Self {
            row,
            col,
            arena: (0..(row * col))
                .map(|_| if req_init { box_muller_f32() } else { 0.0 })
                .collect(),
        }
    }

    pub fn to_matrix(vector: &Vec<u8>) -> Self {
        Self {
            row: 1,
            col: vector.len() - 1,
            arena: vector.iter()
                .skip(1)
                .map(|&x| x as f32/ 255.0) 
                .collect()
        }
    }

    pub fn transpose(&mut self) {
        let temp = self.row;
        self.row = self.col;
        self.col = temp;
    }

    pub fn dot(&self, rhs: &Matrix) -> Matrix {
        if self.col != rhs.row {
            panic!("Illegal dimension: {}x{} * {}x{}", self.row, self.col, rhs.row, rhs.col); 
        }
        
        let mut result = Matrix::new(self.row, rhs.col, false);
        for a in 0..self.row {
            for b in 0..rhs.col {
                let ptr = a * rhs.col + b; 
                for i in 0..self.col {
                    result.arena[ptr] += self.arena[a * self.col + i] * rhs.arena[i * rhs.col + b];
                }
            }
        }
        result
    }

    pub fn add(&self, rhs: &Matrix) -> Matrix {
        if self.row != rhs.row || self.col != rhs.col {
            panic!("Illegal dimension");
        }
        let mut result = Matrix::new(self.row, self.col, false);
        for i in 0..self.arena.len() {
            result.arena[i] = self.arena[i] + rhs.arena[i];
        }
        result
    }

    pub fn backward_relu(mut self, z_mask: &Matrix) -> Self {
        for (error, &z) in self.arena.iter_mut().zip(z_mask.arena.iter()) {
            *error *= relu_derivative_helper(z);
        }
        self
    }

    pub fn get_max(&self) -> (usize, usize) {
        let mut result: (usize, usize) = (0, 0);
        for r in 0..self.row {
            for c in 0..self.col {
                if self.arena[r * self.col + c] > self.arena[result.0 * self.col + result.1] {
                    result = (r, c);
                }
            }
        }
        result
    }
}

impl<'a, 'b> Mul<&'b Matrix> for &'a Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &'b Matrix) -> Self::Output { self.dot(rhs) }
}

impl<'a, 'b> Add<&'b Matrix> for &'a Matrix {
    type Output = Matrix;
    fn add(self, rhs: &'b Matrix) -> Self::Output { self.add(rhs) }
}