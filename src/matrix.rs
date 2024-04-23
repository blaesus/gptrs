use std::ops::{Add, Mul, Sub};
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Add for Matrix {
    type Output = Matrix;

    fn add(self, other: Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) + other.get(i, j));
            }
        }
        result
    }
}

impl Sub for Matrix {
    type Output = Matrix;

    fn sub(self, other: Matrix) -> Matrix {
        self + other.scalar_mul(-1.0)
    }
}

impl Mul for Matrix {
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);

        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
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

}

struct Vector(Matrix);

impl Vector {

    pub fn new(data: Vec<f32>) -> Self {
        Vector(Matrix {
            rows: 1,
            cols: data.len(),
            data,
        })
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.0.data
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
}

impl Mul<f32> for Vector {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut new_data = self.data().clone();
        for i in 0..new_data.len() {
            new_data[i] *= rhs;
        }
        return Self::new(new_data)
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_add() {
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
        let result = a + b;
        assert_eq!(result.get(0, 0), 6.0);
        assert_eq!(result.get(0, 1), 8.0);
        assert_eq!(result.get(1, 0), 10.0);
        assert_eq!(result.get(1, 1), 12.0);
    }

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
    fn test_matrix_sub() {
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
        let result = a - b;
        assert_eq!(result.get(0, 0), -4.0);
        assert_eq!(result.get(0, 1), -4.0);
        assert_eq!(result.get(1, 0), -4.0);
        assert_eq!(result.get(1, 1), -4.0);
    }

    #[test]
    fn test_matrix_dot() {
        let a = Matrix {
            rows: 1,
            cols: 2,
            data: vec![1.0, 2.0],
        };
        let b = Matrix {
            rows: 2,
            cols: 1,
            data: vec![3.0, 4.0],
        };
        let result = a.dot(&b);
        assert_eq!(result, 11.0);
    }

    #[test]
    fn test_matrix_scalar_mul() {
        let a = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let result = a.scalar_mul(2.0);
        assert_eq!(result.get(0, 0), 2.0);
        assert_eq!(result.get(0, 1), 4.0);
        assert_eq!(result.get(1, 0), 6.0);
        assert_eq!(result.get(1, 1), 8.0);
    }
}
