use tensorlib::matrix::Matrix;

pub fn softmax_row(x: Matrix) -> Matrix {
    x.scalar_operation_row(|(row, _)| {
        let max = row.iter().fold(f32::NEG_INFINITY, |acc, &x| x.max(acc));
        row.iter_mut().for_each(|x| *x = (*x - max).exp());
        let sum: f32 = row.iter().sum();
        row.iter_mut().for_each(|x| *x /= sum);
    })
}

pub fn softmax_one_row(mut row: Vec<f32>) -> Vec<f32> {
    let max = row.iter().fold(f32::NEG_INFINITY, |acc, &x| x.max(acc));
    row.iter_mut().for_each(|x| *x = (*x - max).exp());
    let sum: f32 = row.iter().sum();
    row.iter_mut().for_each(|x| *x /= sum);
    row
}

pub fn silu(x: Matrix) -> Matrix {
    x.scalar_operation(|v, _| *v = *v / (1.0 + (-*v).exp()))
}
