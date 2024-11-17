#![allow(unused)]

use plotters::prelude::*;

pub fn plot_loss_to_png(filename: &str, losses: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
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




#[macro_export]
macro_rules! create_network {
    // Cas de base : dernière couche avec OutputLayer
    ($first:expr, $second:expr) => {
        $crate::veclayer::OutputLayer::<$first, $second> {
            layer: $crate::layer::Layer::<$first, $second>::rand(),
        }
    };

    // Cas récursif : capture les deux premiers éléments et continue avec le reste
    ($first:expr, $second:expr, $($rest:expr),+) => {{
        $crate::veclayer::LayerChain::<$first, $second, { create_network!(@last $($rest),+) }, _> {
            layer: $crate::layer::Layer::<$first, $second>::rand(),
            next: create_network!($second, $($rest),+),
        }
    }};

    // Capture le dernier argument de la liste
    (@last $last:expr) => {
        $last
    };

    // Cas récursif avec plusieurs arguments pour capturer la dernière valeur
    (@last $first:expr, $($rest:expr),+) => {
        create_network!(@last $($rest),+)
    };
}


//todo useless avec l'inference
#[macro_export]
macro_rules! network_type {
    // Cas pour un réseau avec uniquement une couche de sortie
    ($inputs:expr, $outputs:expr) => {
        NeuralNetwork<$inputs, $outputs, veclayer::OutputLayer<$inputs, $outputs>, MeanSquaredError<$outputs>>
    };

    // Cas récursif pour plusieurs couches cachées
    ($inputs:expr, $($layers:expr),+) => {{
        // Récupérer le dernier type de couche
        let last_layer = network_type!(@last $($layers),+);
        
        // Construire la chaîne de couches en partant de la couche de sortie
        let layer_chain = network_type!($inputs, $($layers),*); // Étoile pour inclure toutes les couches
        
        // Construire le réseau avec la chaîne de couches
        NeuralNetwork::<$inputs, {last_layer}, {layer_chain}, MeanSquaredError<$inputs>>
    }};

    // Capture le dernier argument de la liste
    (@last $last:expr) => {
        $last
    };

    // Cas récursif avec plusieurs arguments pour capturer la dernière valeur
    (@last $first:expr, $($rest:expr),+) => {
        network_type!(@last $($rest),+)
    };
}









// achi vieux 
// use crate::matrix::Matrix;

// struct Neuron<const INPUTS: usize> {
//     // Un neurone a un poids pour chaque entrée, donc 1 colonne
//     weights: Matrix<INPUTS, 1>,
//     // Le biais est un scalaire
//     bias: f64,
// }
// impl<const INPUTS: usize> Neuron<INPUTS> {
//     fn new() -> Self {
//         Neuron {
//             weights: Matrix::rand(-1.0..1.0),
//             bias: 0.0,
//         }
//     }

//     // z = w * x + b
//     fn forward(&self, inputs: Matrix<INPUTS, 1>) -> Matrix<1, 1> {
//         let weighted_sum = self.weights.transpose().multiply(inputs) + self.bias;
//         weighted_sum.mapv(|x| x.max(0.0)) // ReLU activation
//     }
// }

