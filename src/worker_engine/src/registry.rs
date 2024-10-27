use nn::linear::Module;
use nn::linear_aqlm::LinearAQLM;
use nn::linear_int8::LinearINT8;
use nn::matrix_int8::MatrixInt8;
use std::borrow::Cow;
use std::collections::HashMap;
use tensorlib::matrix::Matrix;

#[derive(Default)]
pub struct LocalLinearRegistry {
    aqlm_storage: HashMap<String, LinearAQLM<'static>>,
    int8_storage: HashMap<String, LinearINT8<'static>>,
}

impl LocalLinearRegistry {
    pub fn echo(&mut self, data: String) -> String {
        data
    }

    pub async fn add_aqlm(
        &mut self,
        name: String,
        codebooks: Vec<f32>,
        scales: Vec<f32>,
        codes: Vec<u8>,
        out_dim: usize,
        in_group_dim: usize,
    ) {
        // info!("add_aqlm {}", name);
        self.aqlm_storage.insert(
            name,
            LinearAQLM::new(
                Cow::Owned(codebooks),
                Cow::Owned(scales),
                Cow::Owned(codes),
                out_dim,
                in_group_dim,
            ),
        );
    }

    pub async fn add_int8(&mut self, name: String, max_values: Vec<f32>, int8_values: Vec<i8>) {
        // info!("add_int8");
        self.int8_storage.insert(
            name,
            LinearINT8::new(MatrixInt8::new(
                Cow::Owned(max_values),
                Cow::Owned(int8_values),
            )),
        );
    }

    pub async fn remove_aqlm(&mut self, name: String) {
        // info!("remove_aqlm {}", name);
        self.aqlm_storage.remove(&name).unwrap();
    }

    pub async fn aqlm_forward(&mut self, name: &str, other: &Matrix<'_>) -> Matrix<'_> {
        // info!("aqlm_forward {}", name);
        let linear = self.aqlm_storage.get_mut(name).unwrap();
        assert_eq!(other.n_rows(), 1);
        linear.forward(other.data()).await
    }

    pub async fn int8_forward(&mut self, name: &str, other: &Matrix<'_>) -> Matrix<'_> {
        // info!("int8_forward");
        let linear = self.int8_storage.get_mut(name).unwrap();
        assert_eq!(other.n_rows(), 1);
        linear.forward(other.data()).await
    }
}
