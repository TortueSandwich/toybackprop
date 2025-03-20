use crate::matrix::Matrix;

#[derive(Clone)]
pub struct LayerBuffer<const INPUTS: usize, const NEURONS: usize> {
    weights_sum: Matrix<NEURONS, INPUTS>,
    biases_sum: Matrix<NEURONS, 1>,
    count: usize,
}

impl<const INPUTS: usize, const NEURONS: usize> Default for LayerBuffer<INPUTS, NEURONS> {
    fn default() -> Self {
        Self {
            weights_sum: Matrix::zeros(),
            biases_sum: Matrix::zeros(),
            count: 0,
        }
    }
}

impl<const INPUTS: usize, const NEURONS: usize> LayerBuffer<INPUTS, NEURONS> {
    /// Accumule les gradients pour les poids et les biais.
    pub fn accumulate_gradients(
        &mut self,
        weight_gradient: Matrix<NEURONS, INPUTS>,
        bias_gradient: Matrix<NEURONS, 1>,
    ) {
        self.count += 1;
        self.weights_sum += weight_gradient;
        self.biases_sum += bias_gradient;
    }

    /// Renvoie les matrices moyennes pondérées pour les poids et les biais.
    pub fn average_gradients(self) -> (Matrix<NEURONS, INPUTS>, Matrix<NEURONS, 1>) {
        if self.count == 0 {
            panic!("Cannot compute average: no gradients accumulated.");
        }
        let scaling_factor = 1.0 / self.count as f64;
        (
            self.weights_sum * scaling_factor,
            self.biases_sum * scaling_factor,
        )
    }
}
