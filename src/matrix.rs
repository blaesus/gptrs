use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};
use crate::rand::{random_f32, random_gaussian};

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;


    fn mul(self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.cols, rhs.rows);
        let mut result = Matrix::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * rhs.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }
}

impl Mul<Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Matrix {
        self * &rhs
    }
}

impl Mul<&Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {
        &self * rhs
    }
}

impl Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Matrix {
        &self * &rhs
    }
}
impl Mul<&Vector> for &Matrix {
    type Output = Vector;

    fn mul(self, rhs: &Vector) -> Vector {
        let vector_as_matrix = Matrix {
            rows: rhs.data().len(),
            cols: 1,
            data: rhs.data().clone(),
        };
        let result_matrix = self * vector_as_matrix;
        Vector(result_matrix)
    }
}

impl Mul<&Vector> for Matrix {
    type Output = Vector;

    fn mul(self, rhs: &Vector) -> Vector {
        &self * rhs
    }
}

impl Mul<Vector> for &Matrix {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Vector {
        self * &rhs
    }
}

impl Mul<Vector> for Matrix {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Vector {
        &self * &rhs
    }
}

impl Mul<f32> for Matrix {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut new_data = self.data.clone();
        for i in 0..new_data.len() {
            new_data[i] *= rhs;
        }
        return Matrix {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        };
    }
}

impl Mul<f32> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f32) -> Self::Output {
        self.clone() * rhs
    }
}

impl Sub for Matrix {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let mut new_data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_data[i * self.cols + j] = self.get(i, j) - rhs.get(i, j);
            }
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }
}

impl AddAssign<Matrix> for Matrix {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.set(i, j, self.get(i, j) + rhs.get(i, j));
            }
        }
    }
}

impl Div<f32> for Matrix {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        let mut new_data = self.data.clone();
        for i in 0..new_data.len() {
            new_data[i] /= rhs;
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }
}

impl SubAssign for Matrix {
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.set(i, j, self.get(i, j) - rhs.get(i, j));
            }
        }
    }
}


impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn new_random(rows: usize, cols: usize) -> Matrix {
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(random_f32());
        }
        Matrix {
            rows,
            cols,
            data,
        }
    }

    pub fn normalize(&mut self) {
        let mean = self.data.iter().sum::<f32>() / self.data.len() as f32;
        let n = self.data.len()-1; // Bessel's correction
        let variance = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let std_dev = variance.sqrt();
        for i in 0..self.data.len() {
            self.data[i] = (self.data[i] - mean) / std_dev;
        }
    }

    pub fn new_kaiming(rows: usize, cols: usize) -> Matrix {
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(random_gaussian(0.0, 2.0 / cols as f32));
        }
        Matrix {
            rows,
            cols,
            data,
        }
    }

    pub fn from_data(data: Vec<f32>, rows: usize, cols: usize) -> Matrix {
        assert_eq!(data.len(), rows * cols);
        Matrix {
            rows,
            cols,
            data,
        }
    }

    fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    fn print(&self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                print!("{:.2} ", self.get(i, j));
            }
            println!();
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut new_data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_data[j * self.rows + i] = self.get(i, j);
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: new_data,
        }
    }

    pub fn to_vector(self) -> Vector {
        assert_eq!(self.cols, 1);
        Vector(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vector(Matrix);

impl Vector {
    pub fn new(data: Vec<f32>) -> Self {
        Vector(Matrix {
            rows: data.len(),
            cols: 1,
            data,
        })
    }

     pub fn zeros(length: usize) -> Self {
        Self::new(vec![0.0; length])
     }

    pub fn new_uniform(element: f32, length: usize) -> Self {
        Self::new((0..length).map(|_| element).collect())
    }

    pub fn new_kaiming(length: usize) -> Self {
        let mut data = Vec::with_capacity(length);
        for _ in 0..length {
            data.push(random_gaussian(0.0, 2.0 / length as f32));
        }
        Self::new(data)
    }

    pub fn normalize(&mut self) {
        let mean = self.data().iter().sum::<f32>() / self.data().len() as f32;
        let n = self.data().len()-1; // Bessel's correction
        let variance = self.data().iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let std_dev = variance.sqrt();
        for i in 0..self.data().len() {
            self.data_mut()[i] = (self.data()[i] - mean) / std_dev;
        }
    }

    pub fn new_random(length: usize) -> Self {
        let mut data = Vec::with_capacity(length);
        for _ in 0..length {
            data.push(random_f32());
        }
        Self::new(data)
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.0.data
    }

    pub fn size(&self) -> usize {
        self.data().len()
    }

    pub fn data_mut(&mut self) -> &mut Vec<f32> {
        &mut self.0.data
    }
    fn dot(&self, other: &Self) -> f32 {
        let data = self.data();
        let other_data = other.data();
        assert_eq!(data.len(), other_data.len());
        data.iter().zip(other_data.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn elementwise_mul(&self, other: &Self) -> Self {
        let data = self.data();
        let other_data = other.data();
        assert_eq!(data.len(), other_data.len());
        let new_data = data.iter().zip(other_data.iter()).map(|(a, b)| a * b).collect();
        Self::new(new_data)
    }

    pub fn as_matrix(&self) -> &Matrix {
        &self.0
    }
}

impl Add<&Vector> for Vector {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        assert_eq!(self.data().len(), rhs.data().len());
        let new_data = self.data().iter().zip(rhs.data().iter()).map(|(a, b)| a + b).collect();
        return Self::new(new_data);
    }
}

impl Add<Vector> for Vector {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl AddAssign<Vector> for Vector {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.data().len(), rhs.data().len());
        for i in 0..self.data().len() {
            self.data_mut()[i] += rhs.data()[i];
        }
    }
}

impl Sub<Vector> for Vector {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl Sub<&Vector> for Vector {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        self + rhs * -1.0
    }
}

impl Sub<&Vector> for &Vector {
    type Output = Vector;

    fn sub(self, rhs: &Vector) -> Self::Output {
        let data = self.data().iter().zip(rhs.data().iter()).map(|(a, b)| a - b).collect();
        Vector::new(data)
    }
}

impl SubAssign<Vector> for Vector {
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.data().len(), rhs.data().len());
        for i in 0..self.data().len() {
            self.data_mut()[i] -= rhs.data()[i];
        }
    }
}

impl Mul<f32> for Vector {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        &self * rhs
    }
}

impl Mul<f32> for &Vector {
    type Output = Vector;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut new_data = self.data().clone();
        for i in 0..new_data.len() {
            new_data[i] *= rhs;
        }
        return Vector::new(new_data);
    }
}

impl Div<f32> for Vector {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        let mut new_data = self.data().clone();
        for i in 0..new_data.len() {
            new_data[i] /= rhs;
        }
        return Vector::new(new_data);
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_mul() {
        let a = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let b = Matrix {
            rows: 2,
            cols: 2,
            data: vec![5.0, 6.0, 7.0, 8.0],
        };
        let result = a * b;
        assert_eq!(result.get(0, 0), 19.0);
        assert_eq!(result.get(0, 1), 22.0);
        assert_eq!(result.get(1, 0), 43.0);
        assert_eq!(result.get(1, 1), 50.0);
    }

    #[test]
    fn test_vector_dot() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), 32.0);
    }

    #[test]
    fn test_vector_mul() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let result = a * 2.0;
        assert_eq!(result.data(), &vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_vector_new_uniform() {
        let a = Vector::new_uniform(1.0, 3);
        assert_eq!(a.data(), &vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_vector_add() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);
        let result = a + b;
        assert_eq!(result.data(), &vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_sub() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);
        let result = a - b;
        assert_eq!(result.data(), &vec![-3.0, -3.0, -3.0]);
    }

    #[test]
    fn test_matrix_vector_product() {
        let a = Matrix {
            rows: 3,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let b = Vector::new(vec![7.0, 8.0]);
        let result = a * b;
        assert_eq!(result.data(), &vec![
            23.0,
            53.0,
            83.0,
        ]);
    }

    #[test]
    fn test_matrix_transpose() {
        let a = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let result = a.transpose();
        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 2);
        assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_vector_normalize() {
        let mut a = Vector::new(vec![1.0, 2.0, 3.0]);
        a.normalize();
        assert_eq!(a.data(), &vec![-1.0, 0.0, 1.0]);
    }
}
