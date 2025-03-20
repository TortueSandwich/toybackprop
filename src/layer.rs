use std::marker::PhantomData;
use std::process::Output;

use num_traits::zero;

use crate::activation::{ActivationFunction, FonctionActivation};
use crate::layerbuffer::LayerBuffer;
use crate::{activation::Sigmoid, matrix::Matrix};



/// NEURONS=Nb of Output
#[derive(Clone)]
pub struct DenseLayer<const INPUTS: usize, const NEURONS: usize> {
    pub weights: Matrix<NEURONS, INPUTS>, // wl
    pub biases: Matrix<NEURONS, 1>, // bl
    pub activation: FonctionActivation,
    gradient_buffer : Option<LayerBuffer<INPUTS, NEURONS>>,
}

impl<const INPUTS: usize, const NEURONS: usize> DenseLayer<INPUTS, NEURONS> {
    pub fn rand() -> Self {
        let weights = Matrix::rand(-0.1..1.0);
        let biases = Matrix::ones()*1.0;
        
        let f = FonctionActivation::ReLU;

        DenseLayer { weights, biases, activation: f, gradient_buffer : None}
    }

    pub fn initialiser_poids_he(&mut self, n : usize) {
        let limit = (2.0 / n as f64).sqrt();
        self.weights = Matrix::rand(-limit..limit);
        self.biases = Matrix::rand(-limit..limit);
    }

    /// a = sig(w * x + b)
    pub fn forward_pass(&self, a: Matrix<INPUTS, 1>) -> Matrix<NEURONS, 1> {
        let weighted_sum: Matrix<NEURONS, 1> = self.weights.clone().multiply(a);
        let t: Matrix<NEURONS, 1> = weighted_sum + self.biases.clone();
        let res = t.mapv(|x| self.activation.activate(x));
        res.mapv(|x| if x.is_nan() {
            panic!("Avertissement: NaN trouvé dans le reseau variable ");
        } else {x});
        res
    }

    /// z = w*x + b
    #[allow(unused)]
    pub fn linear_combination(&self, a: Matrix<INPUTS, 1>) -> Matrix<NEURONS, 1> {
        self.weights.clone().multiply(a) + self.biases.clone()
    }

    pub fn apply_gradients(&mut self, learning_rate : f64) {
        let (w,b) = self.gradient_buffer.take().unwrap().average_gradients();
        let lr:f64 = learning_rate;
        self.weights -= w*lr;
        self.biases -= b*lr;
    }

    // couche quelconque
    #[allow(non_snake_case)]
    pub fn backpropagate(
        &mut self,
        aLprev: Matrix<INPUTS, 1>,
        dCaL: Matrix<NEURONS, 1>,
    ) ->  Matrix<INPUTS, 1> {
        let gradient_buffer = self.gradient_buffer.get_or_insert_with(LayerBuffer::default);

        let zL = self.weights.clone().multiply(aLprev.clone()) + self.biases.clone(); // ok
        let aL = zL.mapv(|x| self.activation.activate(x)); // ok
        // let daLzL = zL.mapv(|x| self.activation.derivative(x)); // ok

        // let dCzL =   daLzL.multiply(dCaL);
        // let dzLwL = aLprev.clone();
        
        let mut wgrad = Matrix::<NEURONS, INPUTS>::zeros();
        for k in 0..INPUTS {
            for j in 0..NEURONS {
                let alk = aL.data[j][0];
                let derivsig=self.activation.derivative(zL.data[j][0]);
                let caj = dCaL.data[j][0];
                wgrad.data[j][k] = alk*derivsig*caj;
            }
        }

        // let dCwL: Matrix<NEURONS, INPUTS> = dzLwL.multiply(dCzL).transpose();
        
        let dzLbL = 1.0;
        let mut bgrad = Matrix::<NEURONS, 1>::zeros();
        for k in 0..NEURONS {
            bgrad.data[k][0] = dzLbL*self.activation.derivative(zL.data[k][0])*dCaL.data[k][0]
        }
        
        gradient_buffer.accumulate_gradients(wgrad, bgrad);

        let mut galprev = Matrix::<INPUTS, 1>::zeros();
        for k in 0..INPUTS {
            let mut acc = 0.0;
            for j in 0..NEURONS {
                let mut wi = 0.0;
                for k2 in 0..INPUTS {
                    wi += aLprev.clone().data[k2][0];
                }
                acc += wi*self.activation.derivative(zL.data[j][0])*dCaL.data[j][0]
            }
            galprev.data[k][0] = acc;
        }

        return galprev;

    }

    #[allow(non_snake_case)]
    pub fn backpropagate_batch<const BATCH_SIZE: usize>(
        &mut self,
        previous_activations: [Matrix<INPUTS, 1>; BATCH_SIZE], // aL
        dCaL_batch: [Matrix<NEURONS, 1>; BATCH_SIZE],  // Gradient pour chaque exemple
    ) -> [Matrix<INPUTS, 1>; BATCH_SIZE] {
        
        let gradient_buffer = self.gradient_buffer.get_or_insert_with(LayerBuffer::default);
        

        
        let mut prev_gradients = std::array::from_fn(|_| Matrix::<INPUTS, 1>::zeros());

        // Pour chaque exemple du batch
        for i in 0..BATCH_SIZE {
            let aLprev = &previous_activations[i];
            let dCaL = &dCaL_batch[i];
    
            // Calculer l'activation et la dérivée pour chaque exemple
            let zL = self.weights.clone().multiply(aLprev.clone()) + self.biases.clone(); // ok
            let aL = zL.mapv(|x| self.activation.activate(x)); // ok
            let daLzL = zL.mapv(|x| self.activation.derivative(x)); // ok
    
            // Gradient par rapport aux poids (wgrad)
            let mut wgrad = Matrix::<NEURONS, INPUTS>::zeros();
            for k in 0..INPUTS {
                for j in 0..NEURONS {
                    let alk = aL.data[j][0];
                    let derivsig = daLzL.data[j][0];
                    let caj = dCaL.data[j][0];
                    wgrad.data[j][k] = alk * derivsig * caj;
                }
            }
    
            // Gradient par rapport aux biais (bgrad)
            let mut bgrad = Matrix::<NEURONS, 1>::zeros();
            for k in 0..NEURONS {
                bgrad.data[k][0] = self.activation.derivative(zL.data[k][0]) * dCaL.data[k][0];
            }
    
            // Mise à jour du gradient du poids et du biais
            gradient_buffer.accumulate_gradients(wgrad, bgrad);
    
            // Calcul du gradient par rapport à l'activation précédente (galprev)
            let mut galprev = Matrix::<INPUTS, 1>::zeros();
            for k in 0..INPUTS {
                let mut acc = 0.0;
                for j in 0..NEURONS {
                    let wi = self.weights.data[j][k];
                    acc += wi * self.activation.derivative(zL.data[j][0]) * dCaL.data[j][0];
                }
                galprev.data[k][0] = acc;
            }
    
            // Enregistrer le gradient pour cet exemple dans la table des gradients
            prev_gradients[i] = galprev;
        }
    
        // Retourner les gradients pour chaque exemple dans le batch
        prev_gradients
    }

    
    pub fn printequation(&self) {
        for n in 0..NEURONS {
            print!("n{n} : ");
            for i in 0..INPUTS {
                print!("{:.5}*x{i} + ", self.weights.data[n][i])
            }
            println!("{:.5}", self.biases.data[n][0]);
        }
    }
}
