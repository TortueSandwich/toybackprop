#![allow(unused)]

use crate::{layer::Layer, matrix::Matrix};

pub trait LayerLink<const INPUTS: usize, const LAST_OUTPUTS: usize> {
    fn forward(&self, input: Matrix<INPUTS, 1>) -> Matrix<LAST_OUTPUTS, 1>;
    // fn backward(&mut self, input: Matrix<INPUTS, 1>, output_gradient: Matrix<LAST_OUTPUTS, 1>) -> Matrix<INPUTS, 1>;
    fn printeq(&self, i :u32);

    fn backward(&mut self, X: &Vec<Matrix<1, INPUTS>> ,  Y: &Vec<Matrix<LAST_OUTPUTS, 1>>); 
}

/// La derniere couche de neurone
/// OUTPUTS = LAST_OUTPUTS
#[derive(Clone)]
pub struct OutputLayer<const INPUTS: usize, const OUTPUTS: usize> {
    pub layer: Layer<INPUTS, OUTPUTS>,
}

impl<const INPUTS: usize, const OUTPUTS: usize> LayerLink<INPUTS, OUTPUTS>
    for OutputLayer<INPUTS, OUTPUTS>
{
    fn forward(&self, input: Matrix<INPUTS, 1>) -> Matrix<OUTPUTS, 1> {
        self.layer.forward(input)
    }
    
    // fn backward(&mut self, input: Matrix<INPUTS, 1>, output_gradient: Matrix<OUTPUTS, 1>) -> Matrix<INPUTS, 1> {
    //     self.layer.backward(input, output_gradient, 1.0) // Learning rate todo
    // }

    fn printeq(&self, i: u32) {
        println!("layer {i}");
        self.layer.printequation();
    }
    fn backward(&mut self, X: &Vec<Matrix<1, INPUTS>> ,  Y: &Vec<Matrix<OUTPUTS, 1>>) {
        self.layer.backward(X, Y);
    }
}

pub struct LayerChain<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const LAST_OUTPUTS: usize,
    Next: LayerLink<OUTPUTS, LAST_OUTPUTS>,
> {
    pub layer: Layer<INPUTS, OUTPUTS>,
    pub next: Next,
}

impl<
        const INPUTS: usize,
        const OUTPUTS: usize,
        const LAST_OUTPUTS: usize,
        Next: LayerLink<OUTPUTS, LAST_OUTPUTS>,
    > LayerLink<INPUTS, LAST_OUTPUTS> for LayerChain<INPUTS, OUTPUTS, LAST_OUTPUTS, Next>
{
    fn forward(&self, input: Matrix<INPUTS, 1>) -> Matrix<LAST_OUTPUTS, 1> {
        let tmp = self.layer.forward(input);
        self.next.forward(tmp)
    }

    // fn backward(&mut self, input: Matrix<INPUTS, 1>, output_gradient: Matrix<LAST_OUTPUTS, 1>) -> Matrix<INPUTS, 1> {
    //     let next_input = self.layer.forward(input.clone());
    //     let next_output_gradient = self.next.backward(next_input, output_gradient);
    //     self.layer.backward(input, next_output_gradient, 1.0) // Learning rate
    // }

    fn printeq(&self,i:u32) {
        println!("layer {i}");
        self.layer.printequation();
        self.next.printeq(i+1);
    }

    fn backward(&mut self, X: &Vec<Matrix<1, INPUTS>> ,  Y: &Vec<Matrix<LAST_OUTPUTS, 1>>) {
        todo!()
    }
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


//todo
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
