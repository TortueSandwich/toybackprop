// use aiframework::matrice::Matrix;

mod tools;  
// mod xor_model;
// mod aiframework; mod xor_matrix;
mod trainning_sets;
mod affine_model;
mod neurongate;
mod threeneurongate;
mod nn;
// mod constant {
//     pub use crate::trainning_sets::*;
//     pub type T = f64;

//     pub const RATE: T = 0.1; 
//     pub const EPS: T = 0.001; //std::f64::EPSILON as T;
//     // pub const TRAIN_PTR : &TrainSet = &XOR_TRAIN;
// }


fn main() {
    nn::run();
}

