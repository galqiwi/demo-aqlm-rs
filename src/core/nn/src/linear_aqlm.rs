use crate::linear::Module;
use async_trait::async_trait;
use std::borrow::Cow;
use tensorlib::functional::linear;
use tensorlib::matrix::{Matrix, OwnedMatrix};

pub struct LinearAQLM<'a> {
    codebooks: Cow<'a, [f32]>,
    scales: Cow<'a, [f32]>,
    codes: Cow<'a, [u8]>,
    out_dim: usize,
    in_group_dim: usize,
}

impl<'a> LinearAQLM<'a> {
    pub fn new(
        codebooks: Cow<'a, [f32]>,
        scales: Cow<'a, [f32]>,
        codes: Cow<'a, [u8]>,
        out_dim: usize,
        in_group_dim: usize,
    ) -> Self {
        assert_eq!(codes.len(), out_dim * in_group_dim * 2);
        Self {
            codebooks,
            scales,
            codes,
            out_dim,
            in_group_dim,
        }
    }
}

#[async_trait(?Send)]
impl Module for LinearAQLM<'_> {
    async fn forward(&mut self, x: &[f32]) -> OwnedMatrix {
        let x = Matrix::from_slice((1, x.len()), x);

        assert_eq!(x.n_rows(), 1);

        let (batch_size, in_dim) = x.shape();
        assert_eq!(in_dim, self.in_group_dim * 8);

        let lut = linear(
            &Matrix::from_slice((batch_size * in_dim / 8, 8), x.data().as_ref()),
            &Matrix::from_slice((2 * 256, 8), &self.codebooks),
        );
        assert_eq!(lut.shape(), (batch_size * in_dim / 8, 2 * 256));

        let mut output = vec![0.0f32; batch_size * self.out_dim];

        aqlm_kernel_0213_120_0123_transp_batch_size_1(
            &mut output,
            lut.data(),
            &self.codes,
            self.out_dim,
            self.in_group_dim,
        );
        let output = OwnedMatrix::from_vec((self.out_dim, batch_size), output).transpose();
        output.multiply_row(&self.scales)
    }

    fn shape(&self) -> (usize, usize) {
        (self.out_dim, self.in_group_dim * 8)
    }
}

/// # Safety
/// Only for use in AQLM kernel
fn aqlm_kernel_0213_120_0123_transp_batch_size_1(
    output: &mut [f32],
    lut: &[f32],
    codes: &[u8],
    out_dim: usize,
    in_group_dim: usize,
) {
    let output_ptr = output.as_mut_ptr();
    let lut_ptr = lut.as_ptr();
    let codes_ptr = codes.as_ptr();

    unsafe {
        for in_grp_idx in 0..in_group_dim {
            for out_idx in 0..out_dim {
                for codebook in 0..2 {
                    let code = *get_codes_120(
                        codes_ptr,
                        in_group_dim,
                        out_dim,
                        codebook,
                        in_grp_idx,
                        out_idx,
                    ) as usize;
                    *output_ptr.add(out_idx) +=
                        *get_lut_0123(lut_ptr, 1, in_group_dim, 0, code, codebook, in_grp_idx);
                }
            }
        }
    }
}

/// # Safety
/// Only for use in AQLM kernel
#[allow(unused)]
#[inline(always)]
pub unsafe fn get_codes_120(
    codes: *const u8,
    in_group_dim: usize,
    out_dim: usize,
    codebook_idx: usize,
    in_group_idx: usize,
    out_idx: usize,
) -> *const u8 {
    codes.add((in_group_idx * (2 * out_dim)) + (codebook_idx * (out_dim)) + out_idx)
}

/// # Safety
/// Only for use in AQLM kernel
#[allow(unused)]
#[inline(always)]
pub unsafe fn get_lut_0123(
    lut: *const f32,
    batch_size: usize,
    in_group_dim: usize,
    batch_idx: usize,
    code_idx: usize,
    codebook_idx: usize,
    in_group_idx: usize,
) -> *const f32 {
    lut.add(
        (batch_idx * (in_group_dim * 2 * 256))
            + (in_group_idx * (2 * 256))
            + (codebook_idx * (256))
            + code_idx,
    )
}
