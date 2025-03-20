#![allow(unused_imports)]

use matrix::Matrix;
use neural_network::NeuralNetwork;
use rand::{rngs::{OsRng, StdRng}, SeedableRng};
use tool::{plot_loss_data, plot_loss_to_png, save_loss_data};
use std::{any::type_name_of_val, sync::Mutex, time::{SystemTime, UNIX_EPOCH}};
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
mod dataset;

use lazy_static::lazy_static;

lazy_static! {
    static ref GLOBAL_RNG: Mutex<StdRng> = Mutex::new(
        StdRng::seed_from_u64(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs())
        // OsRng
        // StdRng::seed_from_u64(42)
    );
}


#[allow(unused)]
fn main() {
    const INPUT_DIM: usize = 2;
    const OUTPUT_DIM: usize = 1;
    
    let network = create_network!(INPUT_DIM, 1, OUTPUT_DIM);

    let mut nn = NeuralNetwork::new(network, loss::FonctionLoss::MSE);
    nn.init_he();

    // XOR inputs et cibles
    // let input = Matrix::from([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
    // let target = Matrix::from([[0.0], [1.0], [1.0], [0.0]]);

    // OR
    let input = [
        Matrix::from([[0.0, 0.0]]),
        Matrix::from([[0.0, 1.0]]),
        Matrix::from([[1.0, 0.0]]),
        Matrix::from([[1.0, 1.0]]),
    ];
    let target = [
        Matrix::from([[0.0]]), 
        Matrix::from([[1.0]]),
        Matrix::from([[1.0]]),
        Matrix::from([[0.0]])
    ];
    // let input = Matrix::from([
        // [0.0, 0.0],
        // [1.0, 0.0],
        // [0.0, 1.0],
        // [1.0, 1.0]
    // ]);
    // let target = Matrix::from([[0.0], [1.0], [1.0], [1.0]]);

    // linéaire 2x+1
    // let input = [
    //     Matrix::from([[0.0]]),
    //     Matrix::from([[1.0]]),
    //     Matrix::from([[2.0]]),
    //     Matrix::from([[3.0]]),
    //     Matrix::from([[4.0]]),
    //     // Matrix::from([[5.0]]),
    // ];
    // let target =  [
    //     Matrix::from([[0.0+1.0]]),
    //     Matrix::from([[2.0+1.0]]),
    //     Matrix::from([[4.0+1.0]]),
    //     Matrix::from([[6.0+1.0]]),
    //     Matrix::from([[8.0+1.0]]),
    //     // Matrix::from([[10.0+10.0]]),
    //     ];




    let mut losses = Vec::new();
    println!("-------- Initial ---------");
    nn.printequations();
    let mut epoch_loss = 0.0;
    for (i, t) in input.clone().into_iter().zip(target.clone().into_iter()) {
        let predicted = nn.predict(i.transpose());
        epoch_loss += nn.loss_function.compute(&predicted,&t);
    }
    println!("loss {}", epoch_loss/input.len() as f64);
    losses.push(epoch_loss/input.len() as f64);
    println!("-------------------------\n");


    

    const EPOCHS: usize = 20;
    const LEARNING_RATE : f64 = 0.02;

    for epoch in 0..EPOCHS {        
        nn.entraine_full_batch(&input.clone(),&target.clone(), LEARNING_RATE);

        let mut epoch_loss = 0.0;

        // dataset 
        for (i, t) in input.clone().into_iter().zip(target.clone().into_iter()) {
            let predicted = nn.predict(i.transpose());
            epoch_loss += nn.loss_function.compute(&predicted,&t);
        }
        
        nn.printequations();
        epoch_loss /= input.len() as f64;
        println!("loss {epoch_loss}\n", );
        // save_loss_data(w, b, epoch_loss, "loss_data.csv");
        losses.push(epoch_loss);
    }

    // plot_loss_data("loss_data.csv");

    println!("-------- finito ---------");
    nn.printequations();

    println!("\nRésultats après l'entraînement ({EPOCHS} epochs) :");
    for i in 0..input.len() {
        let input_example = input[i].clone();
        let predicted = nn.predict(input_example.transpose().clone());
        match input_example.data[0].len()  {
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
    plot_loss_to_png("./loss_plot.png", &losses, EPOCHS).expect("failed to generate pdf");
    println!("finito");
}



