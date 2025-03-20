#![allow(dead_code)]

use std::fmt::{Display, Debug};
use std::ops::Range;
use rand::{distributions::{Standard, Uniform}, prelude::*};
use crate::tools::sigmoid;

#[derive(Clone)]
pub struct Matrix <const M : usize, const N : usize, T>  ([[T;N];M]);

impl<const M : usize, const N : usize, T : Default + Copy> Default for Matrix<M,N,T> {
    fn default() -> Self {
        Matrix([[T::default();N];M])
    }
}

impl<const M: usize, const N: usize, T> Matrix<M, N, T>
where
    T: Default + Copy + rand::distributions::uniform::SampleUniform,
    Standard: Distribution<T>,
{
    pub fn rand() -> Self {
        let mut rng = rand::thread_rng();
        let mut matrix = Matrix::default();

        for i in 0..M {
            for j in 0..N {
                matrix.0[i][j] = rng.gen();
            }
        }

        matrix
    }
    pub fn rand_in_range(r: Range<T>) -> Self {
        let mut rng = rand::thread_rng();
        let mut matrix = Matrix::default();
        let range = Uniform::from(r);
        matrix.0.iter_mut().for_each(|l| {
            l.iter_mut().for_each(|e| {
                *e = range.sample(&mut rng);
            });
        });

        matrix
    }
}
impl<const M : usize, const N : usize, T: Copy + num_traits::real::Real> Matrix<M,N,T> {
    pub fn filled(t:T) -> Self {
        Matrix([[t;N];M])
    }
    pub fn new(m : [[T;N];M]) -> Self {
        Matrix(m)
    }
    pub fn sigmoid(&mut self) -> Self {
        let mut r = self.clone();
        r.0.iter_mut().for_each(|l| l.iter_mut().for_each(|e| *e = sigmoid(e.clone())));
        r
    }
    pub fn get_nb_col_row(&self) -> (usize,usize) {
        (M,N)
    }
    pub fn get(&self, co : (usize,usize)) -> &T {
        &self.0
        .get(co.0).expect("invalid y index")
        .get(co.1).expect("invalid x index")
    }
    pub fn set(&mut self, co : (usize,usize), rhs : T) {
        *(self.0
        .get_mut(co.0).expect("invalid y index")
        .get_mut(co.1).expect("invalid x index"))
         = rhs;
    }
}

impl<const M : usize, const N : usize, T: std::ops::AddAssign<T> + Clone> std::ops::Add<Matrix<M,N,T>> for Matrix<M,N,T> {
    type Output = Self;
    fn add(self, rhs: Matrix<M,N,T>) -> Self::Output {
        let mut r = self.clone();
       for i in 0..M {
           for j in 0..N {
               r.0[i][j] += rhs.0[i][j].clone();
           }
       }
       r
    }
}
impl<const M : usize, const N : usize, T: std::ops::SubAssign<T> + Clone> std::ops::SubAssign<Matrix<M,N,T>> for Matrix<M,N,T> {
    fn sub_assign(&mut self, rhs: Matrix<M,N,T>) {
        for i in 0..M {
           for j in 0..N {
               self.0[i][j] -= rhs.0[i][j].clone();
           }
       }
    }
}

impl<const M : usize, const N : usize, const L : usize, T: Default + Copy + std::ops::AddAssign<T> + std::ops::Mul<T, Output = T>> std::ops::Mul<Matrix<M,N,T>> for Matrix<L,M,T> {
    type Output = Matrix<L,N,T>;
    fn mul(self, rhs: Matrix<M,N,T>) -> Self::Output {
        let mut res: Self::Output = Matrix::default();
        let n = M;
        for i in 0..L {
            for j in 0..N {
                for k in 0..n {
                    res.0[i][j] += self.0[i][k] * rhs.0[k][j]; 
                } 
            }
        }
        res
    }
}
impl<const M : usize, const N : usize, T: Default + Copy + std::ops::AddAssign<T> + std::ops::Mul<T, Output = T>> std::ops::Mul<T> for Matrix<M,N,T> {
    type Output = Matrix<M,N,T>;
    fn mul(self, rhs: T) -> Self::Output {
        let mut res: Self::Output = Matrix::default();
        for i in 0..M {
            for j in 0..N {
                res.0[i][j] += self.0[i][j] * rhs; 
            }
        }
        res
    }
}

impl<const M : usize, const N : usize, T : Display> Debug for Matrix<M,N,T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,"\n")?;
        writeln!(f, "{}", self)?;
        writeln!(f)?;
        // use std::io::Write;
        // use crate::tools::parse_cursor_position;
        // print!("\x1B[6n");
        // std::io::stdout().flush().unwrap();
        // let mut response = String::new();
        // if let Some((row, col)) = parse_cursor_position(&response) {
        //     writeln!(f, "Position du curseur : Ligne {}, Colonne {}", row, col);
        // } else {
        //     panic!("Impossible de récupérer la position du curseur.");
        // }
        Ok(())
    }
}
impl<const M : usize, const N : usize, T : Display> Display for Matrix<M,N,T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if M.eq(&0) || N.eq(&0) { write!(f, "[]")?; return Ok(()); }

//        use std::io::Write;

        write!(f,"\x1b[{}E", M - 1)?;
        let mut y = [0; M];
        for i in 0..M {
            let t = self.0[i][0].to_string();
            y[i] = t.len();
            writeln!(f,"[ {}", t)?;
            //write!(f,"\x1b[{}C", col)?;
        }
        let m = y.iter().max().expect("max pas trouvé");
        write!(f,"\x1b[{}A", M)?;
        write!(f,"\x1b[{}C", m + 4)?;

        for x in 1..N {

            let mut length = [0; M];
            for i in 0..M {
                let t = self.0[i][x].to_string();
                length[i] = t.len(); 
                write!(f, "{} ", t)?;
                write!(f,"\x1b[1B")?;
                write!(f,"\x1b[{}D", t.len() + 1)?;
            }
            let m = length.iter().max().expect("max pas trouvé");
            write!(f,"\x1b[{}A", M)?;
            write!(f,"\x1b[{}C", m)?;

        }
        for _ in 0..M {
            write!(f," ]\x1b[1B\x1b[2D")?;
        }
        write!(f,"\x1b[1A\x1b[2C")?;


        Ok(())
    }
}


impl Into<f64> for Matrix<1,1,f64> {
    fn into(self) -> f64 {
        self.0[0][0]
    }
}

