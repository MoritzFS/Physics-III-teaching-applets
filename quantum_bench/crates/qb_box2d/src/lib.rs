//! Quantum Bench tab "2D box": particle in a box (PS09 38), 2D eigenstates,
//! superpositions, measurement. STUB: to be built by the 2D agent.

use qb_core::{PlaceholderScene, PresetInfo, SceneTab, Tab};

/// The "2D box" tab (the app calls `Box2dTab::new()`).
pub struct Box2dTab;

impl Box2dTab {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> impl Tab {
        SceneTab::new("box2d", "2D box").preset(PresetInfo::new("PS09 38 · Particle in a box").paused(), |_| {
            Box::new(PlaceholderScene::new(
                "PS09 38 · Particle in a box",
                "Under construction: 2D box eigenstates ψ(n₁, n₂), superpositions and their time evolution.",
            ))
        })
    }
}
