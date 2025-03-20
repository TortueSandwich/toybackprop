#![allow(dead_code)]

use std::ops::Range;

use rand::{thread_rng, Rng};

use crate::GLOBAL_RNG;

#[derive(Debug, Clone)]
pub struct Matrix<const ROW: usize, const COL: usize> {
    pub data: [[f64; COL]; ROW],
}

impl<const ROW: usize, const COL: usize> Matrix<ROW, COL> {
    pub fn new(data: [[f64; COL]; ROW]) -> Self {
        Matrix { data }
    }

    pub fn filled(value: f64) -> Self {
        Self::new([[value; COL]; ROW])
    }

    pub fn zeros() -> Self {
        Self::filled(0.0)
    }

    pub fn ones() -> Self {
        Self::filled(1.0)
    }

    pub fn rand(range: Range<f64>) -> Self {
        let mut result = Matrix::<ROW, COL>::new([[0.0; COL]; ROW]);
        for i in 0..ROW {
            for j in 0..COL {
                result.data[i][j] = GLOBAL_RNG.lock().unwrap().gen_range(range.clone());
            }
        }
        result
    }

    pub fn mapv(&self, f: impl Fn(f64) -> f64) -> Self {
        let mut result = Matrix::<ROW, COL>::new([[0.0; COL]; ROW]);
        for i in 0..ROW {
            for j in 0..COL {
                result.data[i][j] = f(self.data[i][j]);
            }
        }
        result
    }

    pub fn mapvinplace(&mut self, f: impl Fn(f64) -> f64) {
        for i in 0..ROW {
            for j in 0..COL {
                self.data[i][j] = f(self.data[i][j]);
            }
        }
    }

    /// |(i,j),x| ...
    pub fn mapvindex(&self, f: impl Fn((usize, usize), f64) -> f64) -> Self {
        let mut result = Matrix::<ROW, COL>::new([[0.0; COL]; ROW]);
        for i in 0..ROW {
            for j in 0..COL {
                result.data[i][j] = f((i, j), self.data[i][j]);
            }
        }
        result
    }

    pub fn transpose(&self) -> Matrix<COL, ROW> {
        let mut result = Matrix::<COL, ROW>::zeros();
        for i in 0..ROW {
            for j in 0..COL {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }
}

impl<const ROW: usize, const INNER: usize> Matrix<ROW, INNER> {
    pub fn multiply<const COL: usize>(self, other: Matrix<INNER, COL>) -> Matrix<ROW, COL> {
        let mut result = Matrix::<ROW, COL>::new([[0.0; COL]; ROW]);
        for i in 0..ROW {
            for j in 0..COL {
                for k in 0..INNER {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        result
    }
}

impl<const ROW: usize, const COL: usize> std::ops::Add for Matrix<ROW, COL> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.mapvindex(|(i, j), x| x + rhs.data[i][j])
    }
}

impl<const ROW: usize, const COL: usize> std::ops::Add<f64> for Matrix<ROW, COL> {
    type Output = Self;
    fn add(self, rhs: f64) -> Self::Output {
        self.mapv(|x| x + rhs)
    }
}

impl<const ROW: usize, const COL: usize> std::ops::Add<Matrix<ROW, COL>> for f64 {
    type Output = Matrix<ROW, COL>;
    fn add(self, rhs: Matrix<ROW, COL>) -> Self::Output {
        rhs + self
    }
}

impl<const ROW: usize, const COL: usize> std::ops::AddAssign<Matrix<ROW, COL>> for Matrix<ROW, COL> {
    fn add_assign(&mut self, rhs: Matrix<ROW, COL>) {
        *self = self.clone() + rhs;
    }
}

impl<const ROW: usize, const COL: usize> std::ops::SubAssign<Matrix<ROW, COL>> for Matrix<ROW, COL> {
    fn sub_assign(&mut self, rhs: Matrix<ROW, COL>) {
        *self = self.clone() - rhs;
    }
}

impl<const ROW: usize, const COL: usize> std::ops::Neg for Matrix<ROW, COL> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.mapv(|x| -x)
    }
}

impl<const ROW: usize, const COL: usize> std::ops::Div<f64> for Matrix<ROW, COL> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self::Output {
        self.mapv(|x| x / rhs)
    }
}

impl<const ROW: usize, const COL: usize> std::ops::DivAssign<f64> for Matrix<ROW, COL> {
    fn div_assign(&mut self, rhs: f64) {
        *self = self.clone() / rhs;
    }
}

impl<const ROW: usize, const COL: usize> std::ops::MulAssign<f64> for Matrix<ROW, COL> {
    fn mul_assign(&mut self, rhs: f64) {
        *self = self.clone() * rhs;
    }
}

impl<const ROW: usize, const COL: usize> std::ops::Sub<f64> for Matrix<ROW, COL> {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self::Output {
        self.mapv(|x| x - rhs)
    }
}

impl<const ROW: usize, const COL: usize> std::ops::Sub<Matrix<ROW, COL>> for Matrix<ROW, COL> {
    type Output = Self;
    fn sub(self, rhs: Matrix<ROW, COL>) -> Self::Output {
        self.mapvindex(|(i,j),x| x - rhs.data[i][j])
    }
}


impl<const ROW: usize, const COL: usize> std::ops::Mul<Matrix<ROW, COL>> for f64 {
    type Output = Matrix<ROW, COL>;
    fn mul(self, rhs: Matrix<ROW, COL>) -> Self::Output {
        rhs.mapv(|x| x * self)
    }
}

impl<const ROW: usize, const COL: usize> std::ops::Mul<f64> for Matrix<ROW, COL> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        rhs * self
    }
}

impl<const N:usize> Matrix<N,1> {
    pub fn sum(&self) -> f64 {
        self.data.iter().fold(0.0, |acc, x|acc+x[0])
    }
}

impl<const ROW: usize, const COL: usize> std::fmt::Display for Matrix<ROW, COL> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_width = self
            .data
            .iter()
            .flat_map(|row| row.iter())
            .map(|x| format!("{:.1}", x).len())
            .max()
            .unwrap_or(0);

        for i in 0..ROW {
            write!(f, "[")?;
            for j in 0..COL {
                write!(f, "{:width$}", self.data[i][j], width = max_width)?;
                if j < COL - 1 {
                    write!(f, ", ")?;
                }
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

impl<const ROW: usize, const COL: usize> From<[[f64; COL]; ROW]> for Matrix<ROW, COL> {
    fn from(value: [[f64; COL]; ROW]) -> Self {
        Matrix::<ROW, COL>::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(matrix.data, [[1.0, 2.0], [3.0, 4.0]]);
    }

    #[test]
    fn test_matrix_zeros() {
        let matrix = Matrix::zeros();
        assert_eq!(matrix.data, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);
    }

    #[test]
    fn test_matrix_ones() {
        let matrix = Matrix::ones();
        assert_eq!(matrix.data, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    }

    #[test]
    fn test_matrix_rand() {
        let matrix = Matrix::<2, 2>::rand(0.0..1.0);
        for row in matrix.data.iter() {
            for &val in row.iter() {
                assert!(val >= 0.0 && val <= 1.0, "value {} is out of range", val);
            }
        }
    }

    #[test]
    fn test_matrix_addition() {
        let m1 = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let m2 = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
        let result = m1 + m2;
        assert_eq!(result.data, [[6.0, 8.0], [10.0, 12.0]]);
    }

    #[test]
    fn test_matrix_add_scalar() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let result = m + 1.0;
        assert_eq!(result.data, [[2.0, 3.0], [4.0, 5.0]]);
    }

    #[test]
    fn test_matrix_multiply() {
        let m1 = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let m2 = Matrix::new([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
        let result = m1.multiply(m2);
        assert_eq!(result.data, [[58.0, 64.0], [139.0, 154.0]]);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = m.transpose();
        assert_eq!(result.data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    }

    #[test]
    fn test_matrix_mapv() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let result = m.mapv(|x| x * 2.0);
        assert_eq!(result.data, [[2.0, 4.0], [6.0, 8.0]]);
    }

    #[test]
    fn test_matrix_mapvindex() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let result = m.mapvindex(|(i, j), x| x + (i + j) as f64);
        assert_eq!(result.data, [[1.0, 3.0], [4.0, 6.0]]);
    }
}

