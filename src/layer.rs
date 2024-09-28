use crate::matrix::Matrix;

/// NEURONS=Nb of Output
pub struct Layer<const INPUTS: usize, const NEURONS: usize> {
    weights: Matrix<NEURONS, INPUTS>,
    biases: Matrix<NEURONS, 1>,
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
        t.mapv(|x| x.max(0.0)) // ReLU activation
    }
}
