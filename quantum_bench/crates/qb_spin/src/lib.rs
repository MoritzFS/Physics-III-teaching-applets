//! Quantum Bench tab "Spin": spin ½, Stern–Gerlach sequences, Ramsey
//! (PS09 35, PS13). STUB: to be built by the spin agent.

use qb_core::{PlaceholderScene, PresetInfo, SceneTab, Tab};

/// The "Spin" tab (the app calls `SpinTab::new()`).
pub struct SpinTab;

impl SpinTab {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> impl Tab {
        SceneTab::new("spin", "Spin").preset(PresetInfo::new("PS13 50 · Stern–Gerlach sequence").paused(), |_| {
            Box::new(PlaceholderScene::new(
                "PS13 50 · Stern–Gerlach sequence",
                "Under construction: Bloch sphere, Stern–Gerlach boxes, Ramsey sequence.",
            ))
        })
    }
}
