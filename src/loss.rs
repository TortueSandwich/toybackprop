use crate::matrix::Matrix;


#[derive(Clone)]
#[allow(unused)]
pub enum FonctionLoss<const OUTPUTS:usize> {
    MSE,
    RMSE,
    // MSLE,
    CrossEntropy
}

impl<const OUTPUTS:usize> FonctionLoss<OUTPUTS> {
    pub fn compute(&self, ŷ: &Matrix<OUTPUTS, 1>, y: &Matrix<OUTPUTS, 1>) -> f64 {
        match self {
            Self::MSE => {
                let diff = ŷ.clone() - y.clone();
                let squared_diff = diff.mapv(|x| x.powi(2));
                squared_diff.sum() / 2.0
            }
            Self::RMSE => Self::MSE.compute(ŷ, y).sqrt(), 
            Self::CrossEntropy => {
                let epsilon = 1e-9; // Pour éviter les log(0)
                let log_probs: Matrix<OUTPUTS, 1> = ŷ.mapv(|x| (x + epsilon).ln());
                let target_probs: Matrix<OUTPUTS, 1> = y.clone();
                
                let product = log_probs.mapv(|x| x * target_probs.data[0][0]);
            
                -product.sum() 
            }
        }
    }

    pub fn compute_grad(&self, ŷ: &Matrix<OUTPUTS, 1>, y: &Matrix<OUTPUTS, 1>) -> Matrix<OUTPUTS, 1> {
        match self {
            Self::MSE => {
                let diff = ŷ.clone() - y.clone(); 
                diff
            }
            Self::RMSE => {
                // let mse_grad = Self::MSE.compute_grad(predicted, target);
                // mse_grad / (2.0 * mse_grad.abs())
                todo!()
            }
            Self::CrossEntropy => {
                let epsilon = 1e-9; // Pour éviter la division par zéro
                let target_probs = y.clone();
                let predicted_probs = ŷ.mapv(|x| if x == 0.0 { epsilon } else { x });
                let mut res = Matrix::<OUTPUTS, 1>::zeros();
                for i in 0..OUTPUTS {
                    res.data[i][0] = -target_probs.data[i][0] / predicted_probs.data[i][0];
                }
                res
            }
        }
    }
}

// #[derive(Clone)]
// pub struct MeanSquaredError<const OUTPUTS: usize>;

// impl<const OUTPUTS: usize> LossFunction<OUTPUTS> for MeanSquaredError<OUTPUTS> {
//     fn compute(&self, predicted: &Matrix<OUTPUTS, 1>, target: &Matrix<OUTPUTS, 1>) -> f64 {
//         let diff = predicted.clone() - target.clone();
//         let squared_diff = diff.mapv(|x| x.powi(2));
//         squared_diff.sum() / 2.0
//     }

//     fn gradient(&self, predicted: &Matrix<OUTPUTS, 1>, target: &Matrix<OUTPUTS, 1>) -> Matrix<OUTPUTS, 1> {
//         predicted.clone() - target.clone()
//     }
// }
