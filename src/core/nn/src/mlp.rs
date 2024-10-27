use crate::functional::silu;
use crate::linear::Module;

pub struct MLPSubmodules<LinearType>
where
    LinearType: Module,
{
    pub up_proj: LinearType,
    pub gate_proj: LinearType,
    pub down_proj: LinearType,
}

pub struct MLP<LinearType>
where
    LinearType: Module,
{
    submodules: MLPSubmodules<LinearType>,
}

impl<LinearType> MLP<LinearType>
where
    LinearType: Module,
{
    pub fn new(submodules: MLPSubmodules<LinearType>) -> Self {
        Self { submodules }
    }
}

impl<LinearType> MLP<LinearType>
where
    LinearType: Module,
{
    pub async fn forward(&mut self, x: &[f32]) -> Vec<f32> {
        // let x = Matrix::from_slice((1, x.len()), x);

        let x = {
            let gate = silu(self.submodules.gate_proj.forward(x).await);
            self.submodules.up_proj.forward(x).await.multiply(&gate)
        }
        .into_data()
        .into_owned();

        // let x = Matrix::from_vec((1, x.len()), x);

        self.submodules
            .down_proj
            .forward(&x)
            .await
            .into_data()
            .into_owned()
    }
}
