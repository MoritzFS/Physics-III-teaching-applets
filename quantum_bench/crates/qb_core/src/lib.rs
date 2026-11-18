//! qb_core: the shared foundation of Quantum Bench. FROZEN after the
//! foundation: tab crates use it, they do not change it.
//!
//! * [`Tab`], [`Ctx`], [`PresetInfo`], [`Scene`], [`SceneTab`]: what a tab implements;
//! * [`run_app`]: the shell (top bar, presets, transport, notes, keys,
//!   screenshot mode on the desktop, URL-hash deep links on the web);
//! * [`num`]: 1D numerics (ħ = m = 1);
//! * [`widgets`]: plots and visual elements shared by all tabs;
//! * [`gpu`]: custom wgpu drawing inside egui (render pipelines only, WebGL2-safe);
//! * [`Look`]: colours, fonts and presentation-mode scaling;
//! * [`QbRng`]: seedable RNG; [`par`]: rayon on the desktop, sequential on the web.
//!
//! See DEVELOPING.md in the workspace root for the guide.

mod clock;
mod fonts;
pub mod gpu;
mod look;
pub mod num;
pub mod par;
pub mod rng;
mod shell;
#[cfg(not(target_arch = "wasm32"))]
mod shot;
mod tab;
#[cfg(target_arch = "wasm32")]
mod web;
pub mod widgets;

pub use clock::wall_time;
pub use look::Look;
pub use rng::QbRng;
pub use shell::{AppConfig, notes_ui, run_app, run_app_with};
pub use tab::{Ctx, PlaceholderScene, PresetInfo, Scene, SceneTab, Tab};

// re-exports, so that tab crates use exactly the same versions
pub use bytemuck;
pub use eframe;
pub use eframe::egui;
pub use eframe::egui_wgpu;
pub use eframe::egui_wgpu::wgpu;
pub use glam;
pub use num_complex;
pub use rustfft;
pub use web_time;
/// `Instant` that works on the desktop and in the browser (never use `std::time::Instant`).
pub use web_time::Instant;

pub use num::{C64, c64};
