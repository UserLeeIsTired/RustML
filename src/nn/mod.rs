pub mod matrix;
pub mod layer;
pub mod activation;
pub mod sequential;
pub mod data;

pub use matrix::Matrix;
pub use layer::Layer;
pub use activation::{
    relu, 
    relu_derivative_helper, 
    softmax,
    cross_entropy_softmax_differentiate
};
pub use sequential::Sequential;
