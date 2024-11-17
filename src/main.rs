#![allow(unused_imports)]

use loss::{LossFunction, MeanSquaredError};
use matrix::Matrix;
use neural_network::NeuralNetwork;
use tool::plot_loss_to_png;
use std::any::type_name_of_val;
use veclayer::{LayerChain, LayerLink};


mod activation;
mod layer;
mod loss;
mod matrix;
mod neural_network;
mod veclayer;
mod tool;
mod matrixchain;
mod layerbuffer;

fn main() {
    const INPUT_DIM: usize = 2;
    const OUTPUT_DIM: usize = 1;
    
    let network = create_network!(INPUT_DIM, 1, OUTPUT_DIM);

    let mut nn = NeuralNetwork::new(network, MeanSquaredError);

    // XOR inputs et cibles
    // let input = Matrix::from([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
    // let target = Matrix::from([[0.0], [1.0], [1.0], [0.0]]);

    // OR
    let input = vec![
        Matrix::from([[0.0, 0.0]]),
        Matrix::from([[0.0, 1.0]]),
        Matrix::from([[1.0, 0.0]]),
        Matrix::from([[1.0, 1.0]]),
    ];
    let target = vec![
        Matrix::from([[0.0]]), 
        Matrix::from([[1.0]]),
        Matrix::from([[1.0]]),
        Matrix::from([[1.0]])
        ];
    // let input = Matrix::from([
        // [0.0, 0.0],
        // [1.0, 0.0],
        // [0.0, 1.0],
        // [1.0, 1.0]
    // ]);
    // let target = Matrix::from([[0.0], [1.0], [1.0], [1.0]]);

    // linéaire
    // let input = vec![
    //     Matrix::from([[0.0]]),
    //     Matrix::from([[1.0]]),
    //     Matrix::from([[2.0]]),
    //     Matrix::from([[3.0]]),
    // ];
    // let target =  vec![
    //     Matrix::from([[0.0]]),
    //     Matrix::from([[2.0]]),
    //     Matrix::from([[4.0]]),
    //     Matrix::from([[6.0]]),
    //     ];

    nn.printequations();


    
    // nn.get_gradiant(input, target);

    

    let epochs = 50;
    let mut loss = [0.0,0.0,0.0,0.0];
    let mut losses = Vec::new();

    for epoch in 0..epochs {
        println!("{epoch}");
        nn.entraine(&input.clone(),&target.clone());
        let mut epoch_loss = 0.0;

        // dataset 
        for (i, t) in input.clone().into_iter().zip(target.clone().into_iter()) {
            let predicted = nn.predict(i.transpose());
            epoch_loss += nn.loss_function.compute(&predicted,&t);
        }
        
        nn.printequations();
        println!("loss {}\n", epoch_loss/input.len() as f64)
    }

    nn.printequations();
    println!("finito");

    println!("\nRésultats après l'entraînement :");
    for i in 0..input.len() {
        let input_example = input[i].clone();
        let predicted = nn.predict(input_example.transpose().clone());
        match predicted.data[0].len()  {
            1 =>println!(
                "Input: ({}), Predicted: {:.4}, Target: {}",
                input_example.data[0][0],
                predicted.data[0][0],
                target[i]
            ), 
            2 =>println!(
                "Input: ({}, {}), Predicted: {:.4}, Target: {}",
                input_example.data[0][0],
                input_example.data[0][1],
                predicted.data[0][0],
                target[i]
            ),
            _ => panic!("wtf"),
        }
        ;
    }
    plot_loss_to_png("./loss_plot.png", &losses).expect("failed to generate pdf");
    println!("finito");
}



