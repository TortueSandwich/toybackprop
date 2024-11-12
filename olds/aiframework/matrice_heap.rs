#![allow(dead_code)]

use std::fmt::{Display, Debug};
use std::ops::Range;
use rand::{distributions::{Standard, Uniform}, prelude::*};
use crate::tools::sigmoid;

#[derive(Clone)]
pub struct MatrixVec<T>(Vec<Vec<T>>);

impl<T: Default> Default for MatrixVec<T> {
    fn default() -> Self {
        MatrixVec(vec![])
    }
}

impl<T> MatrixVec<T>
where
    T: Default + rand::distributions::uniform::SampleUniform,
    Standard: Distribution<T>,
{
    pub fn rand(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut matrixVec = MatrixVec::default();

        for _ in 0..rows {
            let row = (0..cols).map(|_| rng.gen()).collect();
            matrixVec.0.push(row);
        }

        matrixVec
    }

    pub fn rand_in_range(rows: usize, cols: usize, r: Range<T>) -> Self {
        let mut rng = rand::thread_rng();
        let range = Uniform::from(r);
        let mut matrixVec = MatrixVec::default();

        for _ in 0..rows {
            let row = (0..cols).map(|_| range.sample(&mut rng)).collect();
            matrixVec.0.push(row);
        }

        matrixVec
    }
}

impl<T: Default + num_traits::real::Real> MatrixVec<T> {
    pub fn filled(rows: usize, cols: usize, t: T) -> Self {
        let mut matrixVec = MatrixVec::default();

        for _ in 0..rows {
            let row = (0..cols).map(|_| t).collect();
            matrixVec.0.push(row);
        }

        matrixVec
    }

    pub fn new(m: Vec<Vec<T>>) -> Self {
        MatrixVec(m)
    }

    pub fn sigmoid(&mut self) {
        self.0.iter_mut().for_each(|row| {
            row.iter_mut().for_each(|e| *e = sigmoid(*e));
        });
    }

    pub fn get_nb_col_row(&self) -> (usize, usize) {
        (self.0.len(), if self.0.is_empty() { 0 } else { self.0[0].len() })
    }

    pub fn get(&self, co: (usize, usize)) -> Option<&T> {
        self.0.get(co.0).and_then(|row| row.get(co.1))
    }

    pub fn set(&mut self, co: (usize, usize), rhs: T) {
        if let Some(row) = self.0.get_mut(co.0) {
            if let Some(e) = row.get_mut(co.1) {
                *e = rhs;
            }
        }
    }
}

impl<T> std::ops::Add<MatrixVec<T>> for MatrixVec<T>
where
    T: std::ops::AddAssign<T> + Copy,
{
    type Output = Self;

    fn add(mut self, rhs: MatrixVec<T>) -> Self::Output {
        for (row1, row2) in self.0.iter_mut().zip(rhs.0.iter()) {
            for (e1, e2) in row1.iter_mut().zip(row2.iter()) {
                *e1 += *e2;
            }
        }
        self
    }
}

impl<T> std::ops::SubAssign<MatrixVec<T>> for MatrixVec<T>
where
    T: std::ops::SubAssign<T> + Copy,
{
    fn sub_assign(&mut self, rhs: MatrixVec<T>) {
        for (row1, row2) in self.0.iter_mut().zip(rhs.0.iter()) {
            for (e1, e2) in row1.iter_mut().zip(row2.iter()) {
                *e1 -= *e2;
            }
        }
    }
}

// impl<T> std::ops::Mul<MatrixVec<T>> for MatrixVec<T>
// where
//     T: Default + Copy + std::ops::AddAssign<T> + std::ops::Mul<T, Output = T>,
// {
//     type Output = MatrixVec<T>;

//     fn mul(self, rhs: MatrixVec<T>) -> Self::Output {
//         let mut res = MatrixVec::default();
//         let n = self.0[0].len();

//         for row in &self.0 {
//             let mut new_row = vec![T::default(); rhs.0[0].len()];

//             for (i, val) in new_row.iter_mut().enumerate() {
//                 for (j, e) in row.iter().enumerate() {
//                     new_row[i] += e.clone() * rhs.0[j][i].clone();
//                 }
//             }

//             res.0.push(new_row);
//         }

//         res
//     }
// }

impl<T> std::ops::Mul<T> for MatrixVec<T>
where
    T: Default + Copy + std::ops::AddAssign<T> + std::ops::Mul<T, Output = T>,
{
    type Output = MatrixVec<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut res = MatrixVec::default();

        for row in self.0 {
            let new_row = row.into_iter().map(|e| e * rhs).collect();
            res.0.push(new_row);
        }

        res
    }
}

impl<T: Display> Debug for MatrixVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\n")?;
        writeln!(f, "{}", self)?;
        writeln!(f)?;

        Ok(())
    }
}

impl<T: Display> Display for MatrixVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() || self.0[0].is_empty() {
            write!(f, "[]")?;
            return Ok(());
        }

        let M = self.0.len();
        let N = self.0[0].len();

        write!(f, "\x1b[{}E", M - 1)?;

        let mut y = vec![0; M];

        for i in 0..M {
            let t = self.0[i][0].to_string();
            y[i] = t.len();
            writeln!(f, "[ {}", t)?;

            // write!(f,"\x1b[{}C", col)?;
        }

        let m = *y.iter().max().expect("max pas trouvé");
        write!(f, "\x1b[{}A", M)?;
        write!(f, "\x1b[{}C", m + 4)?;

        for x in 1..N {
            let mut length = vec![0; M];

            for i in 0..M {
                let t = self.0[i][x].to_string();
                length[i] = t.len();
                write!(f, "{} ", t)?;
                write!(f, "\x1b[1B")?;
                write!(f, "\x1b[{}D", t.len() + 1)?;
            }

            let m = *length.iter().max().expect("max pas trouvé");
            write!(f, "\x1b[{}A", M)?;
            write!(f, "\x1b[{}C", m)?;
        }

        for _ in 0..M {
            write!(f, " ]\x1b[1B\x1b[2D")?;
        }

        write!(f, "\x1b[1A\x1b[2C")?;

        Ok(())
    }
}

// impl Into<f64> for MatrixVec<1, 1, f64> {
//     fn into(self) -> f64 {
//         self.0[0][0]
//     }
// }
