mod api;
mod loader;
mod log;

use wasm_bindgen::prelude::*;

use crate::log::setup_logger;

#[wasm_bindgen(start)]
pub fn start() {
    setup_logger();
}
