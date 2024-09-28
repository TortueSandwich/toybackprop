use matrix::Matrix;
use neural_network::NeuralNetwork;
use std::any::type_name_of_val;

mod layer;
mod matrix;
mod neural_network;
mod veclayer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nn = NeuralNetwork::new(create_network!(3, 2, 1, 2, 3, 4, 5));
    println!("3, 2, 1, 2, 3, 4, 5\n{}", type_name_of_val(&nn));
    let t = Matrix::rand(0.0..1.0);
    println!("{}", t);
    let res = nn.forward(t);
    println!("{}", res);

    // let network = create_network!(3, 5, 4, 2, 1);
    // println!("3, 5, 4, 2, 1\n{}", type_name_of_val(&network));

    Ok(())
}
