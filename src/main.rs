use loss::{LossFunction, MeanSquaredError};
use matrix::Matrix;
use neural_network::NeuralNetwork;
use std::any::type_name_of_val;
use veclayer::{LayerChain, LayerLink};
use plotters::prelude::*;

mod activation;
mod layer;
mod loss;
mod matrix;
mod neural_network;
mod veclayer;

fn main() {
    const INPUT_DIM: usize = 2;
    const OUTPUT_DIM: usize = 1;
    
    let network = create_network!(INPUT_DIM, 2, OUTPUT_DIM);

    let mut nn = NeuralNetwork::new(network, MeanSquaredError);

    // XOR inputs et cibles
    // let input = Matrix::from([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
    // let target = Matrix::from([[0.0], [1.0], [1.0], [0.0]]);

    // and
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

    

    let epochs = 500;
    let mut loss = [0.0,0.0,0.0,0.0];
    let mut losses = Vec::new();

    for epoch in 0..epochs {
        nn.ducoupppp(&input.clone(),&target.clone());
        let mut epoch_loss = 0.0;
        for (i, t) in input.clone().into_iter().zip(target.clone().into_iter()) {

            epoch_loss += nn.loss_function.compute(&nn.predict(i.transpose()),&t);
        }
        // nn.printequations();
        println!("loss {}", epoch_loss/input.len() as f64)
        // for i in 0..input.data.len() {
            // let input_example = Matrix::from([input.data[i]]).transpose();
        //     let target_example = Matrix::from([target.data[i]]);

        //     nn.fit(input_example.clone(), target_example.clone());

        //     let predicted = nn.predict(input_example.clone());
        //     let new_loss = nn.loss_function.compute(&predicted, &target_example);
        //     loss[i] = new_loss;
        //     epoch_loss += new_loss;
        // }
        // nn.fit(input.transpose(), target);
        // nn.printequations();
        // let average_loss = epoch_loss / input.data.len() as f64;
        // losses.push(average_loss);
        // print!("[");
        // let mut yop = loss.iter();
        // print!("{:.4}", yop.next().unwrap());
        // yop.for_each(|x| print!(", {x:.4}"));
        // println!("] mean = {:.3}", loss.iter().sum::<f64>()/input.data.len() as f64);
    }

    // println!("\nRésultats après l'entraînement :");
    // for i in 0..input.data.len() {
    //     let input_example = Matrix::from([input.data[i]]).transpose();
    //     let predicted = nn.predict(input_example.clone());
    //     println!(
    //         "Input: ({}), Predicted: {:.4}, Target: {}",
    //         input_example.data[0][0],
    //         // input_example.data[1][0],
    //         predicted.data[0][0],
    //         target.data[i][0]
    //     );
    // }

    // let t= nn.network;
    // let w = t.layer.weights.data[0][0];
    // let b = t.layer.biases.data[0][0];
    // println!("w*x + b = {w}*x + {b}");
    plot_loss_to_png("./loss_plot.png", &losses).expect("failed to generate pdf");
    println!("finito");
}



fn plot_loss_to_png(filename: &str, losses: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_loss = losses.iter().cloned().fold(0. / 0., f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Loss over Epochs", ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..losses.len(), 0.0..max_loss)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        losses.iter().enumerate().map(|(x, y)| (x, *y)),
        &RED,
    ))?
    .label("Loss")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}