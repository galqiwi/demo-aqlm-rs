use crate::matrix_serde::SerdeMatrix;
use crate::registry::LocalLinearRegistry;
use serde::{Deserialize, Serialize};
use speedy::{Readable, Writable};
use std::borrow::Cow;
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
#[derive(Default)]
pub struct RPCLinearRegistryServer {
    inner: LocalLinearRegistry,
}

#[wasm_bindgen]
impl RPCLinearRegistryServer {
    pub fn wasm_new() -> Self {
        RPCLinearRegistryServer::default()
    }
}

#[wasm_bindgen]
impl RPCLinearRegistryServer {
    pub async fn serve_serialized(&mut self, request: &[u8]) -> Vec<u8> {
        let request: Request = Request::read_from_buffer(request).unwrap();
        let response = self.serve(request).await;
        response.write_to_vec().unwrap()
    }
}

impl RPCLinearRegistryServer {
    async fn serve(&mut self, request: Request<'_>) -> Response {
        match request {
            Request::AddAQLMRequest(request) => {
                self.serve_add_aqlm(request).await;
                Response::AddAQLMResponse
            }
            Request::AddINT8Request(request) => {
                self.serve_add_int8(request).await;
                Response::AddINT8Response
            }
            Request::RemoveAQLMRequest(request) => {
                self.serve_remove_aqlm(request).await;
                Response::RemoveAQLMResponse
            }
            Request::AQLMForwardRequest(request) => {
                Response::AQLMForwardResponse(self.serve_aqlm_forward(request).await)
            }
            Request::INT8ForwardRequest(request) => {
                Response::INT8ForwardResponse(self.serve_int8_forward(request).await)
            }
        }
    }

    async fn serve_add_aqlm(&mut self, request: AddAQLMRequest<'_>) {
        self.inner
            .add_aqlm(
                request.name,
                request.codebooks.into_owned(),
                request.scales.into_owned(),
                request.codes.into_owned(),
                request.out_dim,
                request.in_group_dim,
            )
            .await;
    }

    async fn serve_add_int8(&mut self, request: AddINT8Request<'_>) {
        self.inner
            .add_int8(
                request.name,
                request.max_values.into_owned(),
                request.int8_values.into_owned(),
            )
            .await;
    }

    async fn serve_remove_aqlm(&mut self, request: RemoveAQLMRequest) {
        self.inner.remove_aqlm(request.name).await;
    }

    async fn serve_aqlm_forward(&mut self, request: AQLMForwardRequest<'_>) -> AQLMForwardResponse {
        let output = self
            .inner
            .aqlm_forward(&request.name, &request.other.into())
            .await;
        AQLMForwardResponse {
            output: output.into(),
        }
    }

    async fn serve_int8_forward(&mut self, request: INT8ForwardRequest<'_>) -> INT8ForwardResponse {
        let output = self
            .inner
            .int8_forward(&request.name, &request.other.into())
            .await;
        INT8ForwardResponse {
            output: output.into(),
        }
    }
}

#[derive(Serialize, Deserialize, Readable, Writable)]
pub enum Request<'a> {
    AddAQLMRequest(AddAQLMRequest<'a>),
    AddINT8Request(AddINT8Request<'a>),
    RemoveAQLMRequest(RemoveAQLMRequest),
    AQLMForwardRequest(AQLMForwardRequest<'a>),
    INT8ForwardRequest(INT8ForwardRequest<'a>),
}

#[derive(Serialize, Deserialize, Readable, Writable)]
pub struct AddAQLMRequest<'a> {
    pub name: String,
    pub codebooks: Cow<'a, [f32]>,
    pub scales: Cow<'a, [f32]>,
    pub codes: Cow<'a, [u8]>,
    pub out_dim: usize,
    pub in_group_dim: usize,
}

#[derive(Serialize, Deserialize, Readable, Writable)]
pub struct RemoveAQLMRequest {
    pub name: String,
}

#[derive(Serialize, Deserialize, Readable, Writable)]
pub struct AddINT8Request<'a> {
    pub name: String,
    pub max_values: Cow<'a, [f32]>,
    pub int8_values: Cow<'a, [i8]>,
}

#[derive(Serialize, Deserialize, Readable, Writable)]
pub struct AQLMForwardRequest<'a> {
    pub name: Cow<'a, str>,
    pub other: SerdeMatrix<'a>,
}

#[derive(Serialize, Deserialize, Readable, Writable)]
pub struct INT8ForwardRequest<'a> {
    pub name: Cow<'a, str>,
    pub other: SerdeMatrix<'a>,
}

#[derive(Serialize, Deserialize, Readable, Writable)]
pub enum Response<'a> {
    AddAQLMResponse,
    AddINT8Response,
    RemoveAQLMResponse,
    AQLMForwardResponse(AQLMForwardResponse<'a>),
    INT8ForwardResponse(INT8ForwardResponse<'a>),
}

#[derive(Serialize, Deserialize, Readable, Writable)]
pub struct AQLMForwardResponse<'a> {
    pub output: SerdeMatrix<'a>,
}

#[derive(Serialize, Deserialize, Readable, Writable)]
pub struct INT8ForwardResponse<'a> {
    pub output: SerdeMatrix<'a>,
}
