//! Optics Bench: a GPU ray-optics bench and a 4f Fourier-optics bench.
//!
//! The same app runs as a desktop program (`main.rs`) and in the browser
//! (WebAssembly + WebGPU, also started from `main.rs`).

mod app;
mod configs;
mod fourier;
mod fourier_ui;
mod gpu;
mod scene;
mod trace;
#[cfg(target_arch = "wasm32")]
mod web;

pub use app::OpticsApp;
