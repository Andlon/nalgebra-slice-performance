use nalgebra::{U3, U1, DMatrixSliceMut, Dynamic, DMatrix, MatrixSlice, MatrixSliceMN,
               Matrix3, MatrixMN};

use std::ops::AddAssign;

#[inline(never)]
fn contract(
    output: &mut DMatrixSliceMut<f64>,
    f: &Matrix3<f64>,
    a: &MatrixSliceMN<f64, U3, Dynamic>)
{
    let num_nodes = a.ncols();
    let output_dim = num_nodes * 3;
    assert_eq!(output_dim, output.nrows());
    assert_eq!(output_dim, output.ncols());

    let u = f * a;
    let v = f.transpose() * a;
    let d = 3;
    let mut contraction = Matrix3::zeros();

    // Since output is column-major, loop over (block) columns first
    for j in 0 .. num_nodes {
        for i in 0 .. num_nodes {
            let u_i = u.fixed_slice::<U3, U1>(0, i);
            let u_j = u.fixed_slice::<U3, U1>(0, j);
            let v_i = v.fixed_slice::<U3, U1>(0, i);
            let v_j = v.fixed_slice::<U3, U1>(0, j);

            contraction.ger(1.0, &u_i, &u_j, 0.0);
            contraction.ger(1.0, &v_i, &v_j, 1.0);
            contraction.ger(1.0, &v_j, &v_i, 1.0);

            output
                .fixed_slice_mut::<U3, U3>(i * d, j * d)
                .add_assign(&contraction);
        }
    }
}

fn main() {
    let f = Matrix3::<f64>::new_random();
    let a = MatrixMN::<f64, U3, Dynamic>::from_row_slice(&[
        0.79823853, 0.53879483, 0.6145651 , 0.56647738, 0.80380162, 0.81328391, 0.92302888, 0.81700116, 0.40729593, 0.36585753,
        0.42701343, 0.69061995, 0.90149585, 0.17902489, 0.29973298, 0.8654594 , 0.39307017, 0.33597961, 0.89614737, 0.03698405,
        0.9097741 , 0.90695223, 0.05189938, 0.49869605, 0.32052228, 0.44186043, 0.32517814, 0.16204256, 0.14232612, 0.707076,
    ]);

    let mut output = DMatrix::zeros(30, 30);

    for _ in 0 .. 10000000 {
        contract(&mut DMatrixSliceMut::from(&mut output), &f, &MatrixSlice::from(&a));
    }

    // Use the final output so as to prevent the compiler from discarding our whole program
    println!("Output: {}", output.abs().max());
}