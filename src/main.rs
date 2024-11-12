use loss::{LossFunction, MeanSquaredError};
use matrix::Matrix;
use neural_network::NeuralNetwork;
use std::any::type_name_of_val;
use veclayer::{LayerChain, LayerLink};

mod activation;
mod layer;
mod loss;
mod matrix;
mod neural_network;
mod veclayer;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // type NN = 
    // network_type!(2, 3, 4, 5);
    // println!("{}", std::any::type_name::<NN>());

    // create_static_nn!(nn_static, 2, 1);
    // unsafe {
    //     // Accéder et utiliser le réseau
    //     if let Some(ref mut nn) = nn_static {
    //         // Exemple d'utilisation : faire avancer un input
    //         let input = Matrix::from([[0.0], [0.0]]);
    //         let output = nn.forward(input);
    //         println!("Output: {:?}", output);
    //     }
    // }

    // Create a Neural Network: Input layer (2 inputs) -> Hidden layer (3 neurons) -> Output layer (1 output)
    let mut nn = 
        NeuralNetwork::new(create_network!(8,7,8, 9), MeanSquaredError);

    // // XOR input and target
    // // let input = Matrix::from([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],[1.0, 1.0]]); // Shape: 4x2
    // // let target = Matrix::from([[0.0], [1.0], [1.0], [0.0]]); // Shape: 4x1

    // let input = Matrix::from([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],[1.0, 1.0]]); // Shape: 4x2
    // let target = Matrix::from([[0.0], [1.0], [1.0], [1.0]]); // Shape: 4x1

    // // Training the network
    // let mut loss = 0.0;
    // for _ in 0..1000 {
    //     for i in 0..4 {
    //         let input_example = Matrix::from([[input.data[i][0]], [input.data[i][1]]]); // Selecting the input for XOR
    //         let target_example = Matrix::from([[target.data[i][0]]]); // Corresponding target
    //         nn.train(input_example.clone(), target_example.clone());
    //         if i == 1 {
    //             let predicted = nn.network.forward(input_example.clone());
    //             let newloss = nn.loss_function.compute(&predicted, &target_example);
    //             if newloss < loss {
    //                 print!("\x1b[92m");
    //             } else {
    //                 print!("\x1b[91m");
    //             }
    //             println!("{}\x1b[0m", newloss);
    //             loss = newloss
    //         }
    //     }
    // }

    // // Test the network after training
    // for i in 0..4 {
    //     let input_example = Matrix::from([[input.data[i][0]], [input.data[i][1]]]);
    //     let predicted = nn.forward(input_example);
    //     println!("Input: ({}, {}), Predicted: {}", input.data[i][0], input.data[i][1], predicted.data[0][0]);
    // }

    Ok(())
}
