use crate::registry_rpc::{
    AQLMForwardRequest, AddAQLMRequest, AddINT8Request, INT8ForwardRequest, RemoveAQLMRequest,
    Request, Response,
};
use speedy::{Readable, Writable};
use std::borrow::Cow;
use tensorlib::matrix::{Matrix, OwnedMatrix};
use tokio::sync::mpsc;
use web_sys::Worker;

pub struct RPCLinearRegistryHandle {
    worker: Worker,
    response_rx: mpsc::Receiver<Vec<u8>>,
    response_tx: mpsc::Sender<Vec<u8>>,
}

impl RPCLinearRegistryHandle {
    pub fn new(worker: Worker) -> Self {
        let (response_tx, response_rx) = mpsc::channel(1);

        Self {
            worker,
            response_rx,
            response_tx,
        }
    }
}

impl RPCLinearRegistryHandle {
    pub fn get_sender(&self) -> mpsc::Sender<Vec<u8>> {
        self.response_tx.clone()
    }
}

impl RPCLinearRegistryHandle {
    async fn send_serialized(&mut self, request: Request<'_>) -> Vec<u8> {
        self.worker
            .post_message(&request.write_to_vec().unwrap().into())
            .unwrap();
        self.response_rx.recv().await.unwrap()
    }
}

impl RPCLinearRegistryHandle {
    pub async fn add_aqlm(
        &mut self,
        name: String,
        codebooks: &[f32],
        scales: &[f32],
        codes: &[u8],
        out_dim: usize,
        in_group_dim: usize,
    ) {
        self.send_serialized(Request::AddAQLMRequest(AddAQLMRequest {
            name,
            codebooks: Cow::Borrowed(codebooks),
            scales: Cow::Borrowed(scales),
            codes: Cow::Borrowed(codes),
            out_dim,
            in_group_dim,
        }))
        .await;
    }

    pub async fn add_int8(&mut self, name: String, max_values: &[f32], int8_values: &[i8]) {
        self.send_serialized(Request::AddINT8Request(AddINT8Request {
            name,
            max_values: Cow::Borrowed(max_values),
            int8_values: Cow::Borrowed(int8_values),
        }))
        .await;
    }

    pub async fn remove_aqlm(&mut self, name: String) {
        self.send_serialized(Request::RemoveAQLMRequest(RemoveAQLMRequest { name }))
            .await;
    }

    pub async fn aqlm_forward(&mut self, name: &str, other: &Matrix<'_>) -> OwnedMatrix {
        if let Response::AQLMForwardResponse(response) = Response::read_from_buffer(
            &self
                .send_serialized(Request::AQLMForwardRequest(AQLMForwardRequest {
                    name: Cow::Borrowed(name),
                    other: other.into(),
                }))
                .await,
        )
        .unwrap()
        {
            let output: Matrix = response.output.into();
            return output.into_owned();
        }
        panic!()
    }

    pub async fn int8_forward(&mut self, name: &str, other: &Matrix<'_>) -> OwnedMatrix {
        if let Response::INT8ForwardResponse(response) = Response::read_from_buffer(
            &self
                .send_serialized(Request::INT8ForwardRequest(INT8ForwardRequest {
                    name: Cow::Borrowed(name),
                    other: other.into(),
                }))
                .await,
        )
        .unwrap()
        {
            let output: Matrix = response.output.into();
            return output.into_owned();
        }
        panic!()
    }
}
