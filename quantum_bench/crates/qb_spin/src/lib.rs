//! Quantum Bench tab "Spin": two-state systems and spin ½ (PS09 35, PS13 50, 51).
//!
//! * [`spinor`]: the two-level algebra with the problem sets' conventions (tested);
//! * [`bloch`]: the Bloch sphere widget and the state panel (column vector, P bars);
//! * [`noncommuting`]: PS09 35, measurements of non-commuting observables;
//! * [`stern_gerlach`]: PS13 50, chains of Stern–Gerlach boxes;
//! * [`ramsey`]: PS13 51, Larmor precession and the Ramsey sequence.

pub mod bloch;
pub mod noncommuting;
pub mod ramsey;
pub mod spinor;
pub mod stern_gerlach;
pub mod util;

use qb_core::{PresetInfo, SceneTab, Tab};

use ramsey::{RamseyScene, RamseyVariant};
use stern_gerlach::{SgPreset, SgScene};

/// The "Spin" tab (the app calls `SpinTab::new()`).
pub struct SpinTab;

impl SpinTab {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> impl Tab {
        let mut tab = SceneTab::new("spin", "Spin").preset(PresetInfo::new("PS09 35 · Non-commuting observables").paused(), |_| {
            Box::new(noncommuting::NonCommutingScene::new())
        });
        for p in SgPreset::ALL {
            tab = tab.preset(PresetInfo::new(p.title()), move |_| Box::new(SgScene::new(p)));
        }
        for v in RamseyVariant::ALL {
            let info = PresetInfo::new(v.title());
            let info = if v.autoplay() { info } else { info.paused() };
            tab = tab.preset(info, move |ctx| Box::new(RamseyScene::new(v, ctx.rng)));
        }
        tab
    }
}
