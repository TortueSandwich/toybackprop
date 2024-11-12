use crate::{activation::Sigmoid, matrix::Matrix};
use crate::activation::ActivationFunction;

/// NEURONS=Nb of Output
pub struct Layer<const INPUTS: usize, const NEURONS: usize> {
    pub weights: Matrix<NEURONS, INPUTS>,
    pub biases: Matrix<NEURONS, 1>,
}

impl<const INPUTS: usize, const NEURONS: usize> Layer<INPUTS, NEURONS> {
    pub fn rand() -> Self {
        let weights = Matrix::rand(-1.0..1.0);
        let biases = Matrix::ones();
        Layer { weights, biases }
    }

    /// z = w * a + b
    /// 3x1 = 3x2*2x1 + 3x1
    pub fn forward(&self, a: Matrix<INPUTS, 1>) -> Matrix<NEURONS, 1> {
        let weighted_sum = self.weights.clone().multiply(a);
        let t =  weighted_sum + self.biases.clone();
        let s = Sigmoid;
        t.mapv(|x| s.activate(x))
        // t.mapv(|x| x.max(0.0)) // ReLU activation
    }


    pub fn backward(&mut self, input: Matrix<INPUTS, 1>, output_gradient: Matrix<NEURONS, 1>, learning_rate: f64) -> Matrix<INPUTS, 1> {
        // Gradient des poids: output_gradient * input.transpose()
        let weight_gradient = output_gradient.clone().multiply(input.transpose());
        
        // Mettre à jour les poids et les biais
        self.weights = self.weights.clone() - weight_gradient * learning_rate;
        self.biases = self.biases.clone() - output_gradient.clone() * learning_rate;

        // Gradient de l'entrée: weights.transpose() * output_gradient
        let input_gradient = self.weights.transpose().multiply(output_gradient);
        input_gradient
    }
}
