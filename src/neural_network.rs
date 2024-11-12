use crate::{loss::LossFunction, matrix::Matrix, veclayer::LayerLink};

pub struct NeuralNetwork<const INPUTS: usize, const OUTPUTS: usize, H: LayerLink<INPUTS, OUTPUTS>, L: LossFunction<OUTPUTS>> {
    pub network: H,
    pub loss_function: L,
}

impl<const INPUTS: usize, const OUTPUTS: usize, H: LayerLink<INPUTS,OUTPUTS>, L: LossFunction<OUTPUTS>> NeuralNetwork<INPUTS,OUTPUTS, H, L> {
    pub fn new(network: H, loss_function: L) -> Self {
        NeuralNetwork {
            network,
            loss_function,
        }
    }

    pub fn forward(&self, input: Matrix<INPUTS, 1>) -> Matrix<OUTPUTS, 1> {
        self.network.forward(input) 
    }

    pub fn train(&mut self, input: Matrix<INPUTS, 1>, target: Matrix<OUTPUTS, 1>) {
        // Passer l'entrée à travers le réseau
        let predicted = self.network.forward(input.clone());

        // Calculer la perte
        let loss = self.loss_function.compute(&predicted, &target);

        // Calculer le gradient de la perte
        let output_gradient = self.loss_function.gradient(&predicted, &target);

        // Rétropropagation
        self.network.backward(input, output_gradient);
        
        // println!("Loss: {}", loss);
    }
    
}



// vec Layer<2,3> Layer<3,2>
// vec Layer<A,B> Layer<B,C> Layer<C,D> ...
// reseau qui prend A entrées et D outputs

// z2 = w2*a1+b2
// 3x1 = 3x2*2x1 + 3x1

// z3 = w3*a2+b3
// 2x1 = 2x3*3x1 +2x1