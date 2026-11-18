//! Quantum Bench tab "1D waves": one scene per exercise (PS09 36/37, PS10 38–41, PS12 49).
//!
//! Each preset of the "Examples" menu builds its own [`qb_core::Scene`];
//! [`gaussian::GaussianScene`] (PS10 38) is the reference scene to copy from.
//! Shared pieces live in [`common`], closed-form results in [`physics`].

pub mod barrier;
pub mod common;
pub mod decay;
pub mod draw;
pub mod efield;
pub mod gaussian;
pub mod oscillator;
pub mod physics;
pub mod truncated;

use qb_core::{PresetInfo, SceneTab, Tab};

/// The "1D waves" tab (the app calls `Wave1dTab::new()`).
pub struct Wave1dTab;

impl Wave1dTab {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> impl Tab {
        SceneTab::new("wave1d", "1D waves")
            .preset(PresetInfo::new("PS10 38 · Gaussian wave packet"), |_| {
                Box::new(gaussian::GaussianScene::new())
            })
            .preset(PresetInfo::new("PS10 38 · Moving packet, k₀ = 3 (Re ψ)"), |_| {
                Box::new(gaussian::GaussianScene::moving())
            })
            .preset(PresetInfo::new("PS09 36 · Decaying particle"), |_| {
                Box::new(decay::DecayScene::new())
            })
            .preset(PresetInfo::new("PS09 37 · Truncated plane wave").paused(), |_| {
                Box::new(truncated::TruncatedScene::new())
            })
            .preset(PresetInfo::new("PS10 39 · Barrier tunnelling, E < V"), |_| {
                Box::new(barrier::BarrierScene::new(barrier::BarrierPreset::Tunnelling))
            })
            .preset(PresetInfo::new("PS10 39 · Above the barrier, resonance T = 1"), |_| {
                Box::new(barrier::BarrierScene::new(barrier::BarrierPreset::Resonance))
            })
            .preset(PresetInfo::new("PS10 39 · Above the barrier, off resonance"), |_| {
                Box::new(barrier::BarrierScene::new(barrier::BarrierPreset::OffResonance))
            })
            .preset(PresetInfo::new("PS10 39 · Thick barrier"), |_| {
                Box::new(barrier::BarrierScene::new(barrier::BarrierPreset::Thick))
            })
            .preset(PresetInfo::new("PS10 40 · Oscillator ψ₀ + ψ₁ (sloshing)"), |_| {
                Box::new(oscillator::OscillatorScene::new(oscillator::OscPreset::Sloshing))
            })
            .preset(PresetInfo::new("PS10 40 · Oscillator ψ₀ + ψ₂ (breathing)"), |_| {
                Box::new(oscillator::OscillatorScene::new(oscillator::OscPreset::Breathing))
            })
            .preset(PresetInfo::new("PS10 40 · Oscillator number state ψ₃"), |_| {
                Box::new(oscillator::OscillatorScene::new(oscillator::OscPreset::Number(3)))
            })
            .preset(PresetInfo::new("PS10 40 · Coherent states"), |_| {
                Box::new(oscillator::OscillatorScene::new(oscillator::OscPreset::Coherent))
            })
            .preset(PresetInfo::new("PS10 41 · Draw V: double well"), |_| {
                Box::new(draw::DrawScene::new(draw::DrawPreset::DoubleWell))
            })
            .preset(PresetInfo::new("PS10 41 · Draw V: double barrier"), |_| {
                Box::new(draw::DrawScene::new(draw::DrawPreset::DoubleBarrier))
            })
            .preset(PresetInfo::new("PS10 41 · Draw V: potential step"), |_| {
                Box::new(draw::DrawScene::new(draw::DrawPreset::Step))
            })
            .preset(PresetInfo::new("PS12 49 · Oscillator in an electric field"), |_| {
                Box::new(efield::EfieldScene::new())
            })
    }
}
