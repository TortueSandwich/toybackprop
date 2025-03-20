#![allow(unused)]

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;


use plotters::prelude::*;

pub fn plot_loss_to_png(filename: &str, losses: &[f64], epochs : usize) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_loss = losses.iter().cloned().fold(0. / 0., f64::max);

    let caption =format!("Loss over {epochs} Epochs");
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20).into_font())
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

pub fn save_loss_data(w: f64, b: f64, epoch_loss: f64, file_path: &str) {
    let path = Path::new(file_path);
    let file = if path.exists() {
        File::options().append(true).open(path).unwrap()
    } else {
        File::create(path).unwrap()
    };

    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}, {}, {}", w, b, epoch_loss).unwrap();
}

pub fn plot_loss_data(file_path: &str) {
    // Lire le fichier CSV
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);

    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();
    let mut color_vals = Vec::new();  // Pour stocker l'intensité de la couleur en fonction de la perte

    // Variables pour trouver min et max des valeurs de w et b
    let mut min_w = f64::INFINITY;
    let mut max_w = f64::NEG_INFINITY;
    let mut min_b = f64::INFINITY;
    let mut max_b = f64::NEG_INFINITY;

    // Lire les données et calculer les min et max de w et b
    for line in reader.lines() {
        let line = line.unwrap();
        let parts: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse().unwrap())
            .collect();

        if parts.len() == 3 {
            let w = parts[0];
            let b = parts[1];
            let loss = parts[2];

            x_vals.push(w);
            y_vals.push(b);
            color_vals.push(loss);

            // Calcul des min et max
            if w < min_w { min_w = w; }
            if w > max_w { max_w = w; }
            if b < min_b { min_b = b; }
            if b > max_b { max_b = b; }
        }
    }

    // Créer le fichier de sortie pour le graphique
    let root = BitMapBackend::new("loss_plot_heat.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Perte en fonction des poids et biais", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_w..max_w, min_b..max_b)
        .unwrap();

    // Configurer la grille et les légendes
    chart
        .configure_mesh()
        .x_desc("Poids (w)")
        .y_desc("Biais (b)")
        .x_labels(10)  // Ajouter un certain nombre de labels sur l'axe x
        .y_labels(10)  // Ajouter un certain nombre de labels sur l'axe y
        .draw()
        .unwrap();


    chart
        .draw_series(
            x_vals
                .iter()
                .zip(y_vals.iter())
                .zip(color_vals.iter())
                .map(|((x, y), &loss)| {
                    let color = if loss < 0.1 {
                        RGBAColor(0, 0, 0, 0.75)
                    } else if loss < 1.0 {
                        RGBAColor(255, 255, 0, 0.75)
                    } else {
                        RGBAColor(255, 0, 0, 0.75)
                    };

                    Circle::new((*x, *y), 5, ShapeStyle {
                        color,
                        filled: true,
                        stroke_width: 1,
                    })
                }),
        )
        .unwrap();

    // Sauvegarder l'image
    println!("Graphique généré dans 'loss_plot.png'.");
}

fn main() {
    plot_loss_data("loss_data.csv");
}

#[macro_export]
macro_rules! create_network {
    // Cas de base : dernière couche avec OutputLayer
    ($first:expr, $second:expr) => {
        $crate::veclayer::OutputLayer::<$first, $second> {
            layer: $crate::layer::DenseLayer::<$first, $second>::rand(),
        }
    };

    // Cas récursif : capture les deux premiers éléments et continue avec le reste
    ($first:expr, $second:expr, $($rest:expr),+) => {{
        $crate::veclayer::LayerChain::<$first, $second, { create_network!(@last $($rest),+) }, _> {
            layer: $crate::layer::DenseLayer::<$first, $second>::rand(),
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

