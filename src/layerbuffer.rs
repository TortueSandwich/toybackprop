use crate::matrix::Matrix;


#[derive(Clone)]
pub struct LayerBuffer<const INPUTS: usize, const NEURONS: usize> {
    wbuff : Matrix<NEURONS, INPUTS>,
    bbuff : Matrix<NEURONS, 1>,
    traincount : usize
}

impl<const INPUTS: usize, const NEURONS: usize> Default for LayerBuffer<INPUTS, NEURONS> {
    fn default() -> Self {
        LayerBuffer {
            wbuff : Matrix::zeros(),
            bbuff : Matrix::zeros(),
            traincount : 0
        }
    }
}

impl<const INPUTS: usize, const NEURONS: usize> LayerBuffer<INPUTS, NEURONS> {
    pub fn add_weight_grad(&mut self,  matw : Matrix<NEURONS, INPUTS>, matb : Matrix<NEURONS, 1>) {
        self.traincount += 1;
        self.wbuff = self.wbuff.clone() + matw;
        self.bbuff = self.bbuff.clone() + matb;
    }

    pub fn get_mats(self) -> (Matrix<NEURONS, INPUTS>,Matrix<NEURONS, 1>) {
        let pond = 1.0/self.traincount as f64;
        return (self.wbuff*pond, self.bbuff*pond) ;
    }
}