use crate::matrix::Matrix;

pub trait LossFunction<const OUTPUTS:usize> {
    fn compute(&self, predicted: &Matrix<OUTPUTS, 1>, target: &Matrix<OUTPUTS, 1>) -> f64;
    fn gradient(&self, predicted: &Matrix<OUTPUTS, 1>, target: &Matrix<OUTPUTS, 1>) -> Matrix<OUTPUTS, 1>;
}

pub struct MeanSquaredError<const OUTPUTS: usize>;

impl<const OUTPUTS: usize> LossFunction<OUTPUTS> for MeanSquaredError<OUTPUTS> {
    fn compute(&self, predicted: &Matrix<OUTPUTS, 1>, target: &Matrix<OUTPUTS, 1>) -> f64 {
        let diff = predicted.clone() - target.clone();
        let squared_diff = diff.mapv(|x| x.powi(2));
        squared_diff.sum() / 2.0 // Retourne la perte
    }

    fn gradient(&self, predicted: &Matrix<OUTPUTS, 1>, target: &Matrix<OUTPUTS, 1>) -> Matrix<OUTPUTS, 1> {
        (predicted.clone() - target.clone()) * (1.0 / OUTPUTS as f64) // Gradient de la perte MSE
    }
}
