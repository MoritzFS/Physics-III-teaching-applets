//! Screenshot mode (desktop only), driven by environment variables:
//!
//! | variable | meaning |
//! |---|---|
//! | `QB_SHOT=out.png` | take a screenshot into this file and quit |
//! | `QB_TAB=<id, name or index>` | tab to show (default: the first) |
//! | `QB_PRESET=<i>` | preset index, 0-based (default 0) |
//! | `QB_T=<t>` | advance the simulation to time t (exact steps, not wall clock) |
//! | `QB_ACTION=a;b@2.5;c` | actions for `Tab::action`; `@t` runs one at time t, the others at QB_T |
//! | `QB_PRESENT=1` | presentation mode |
//! | `QB_LIGHT=1` | light theme |
//! | `QB_SEED=<u64>` | RNG seed (default 1) |
//! | `QB_SIZE=1500x950` | window size in points |
//! | `QB_WAIT=<s>` | extra wall-clock seconds before the shot (GPU warm-up; default 0.5) |
//!
//! The simulation is paused for the shot, so the picture shows exactly time QB_T.

use std::path::PathBuf;

/// The parsed variables.
#[derive(Clone, Debug)]
pub(crate) struct ShotPlan {
    pub path: PathBuf,
    pub tab: Option<String>,
    pub preset: usize,
    pub t: Option<f64>,
    /// (time or None = at QB_T, action)
    pub actions: Vec<(Option<f64>, String)>,
    pub present: bool,
    pub light: bool,
    pub seed: u64,
    pub size: [f32; 2],
    pub wait: f64,
}

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

impl ShotPlan {
    pub fn from_env() -> Option<Self> {
        let path = PathBuf::from(var("QB_SHOT")?);
        let actions = var("QB_ACTION")
            .map(|s| {
                s.split(';')
                    .map(str::trim)
                    .filter(|a| !a.is_empty())
                    .map(|a| match a.rsplit_once('@') {
                        Some((name, t)) if t.trim().parse::<f64>().is_ok() => (Some(t.trim().parse().unwrap()), name.trim().to_string()),
                        _ => (None, a.to_string()),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let size = var("QB_SIZE")
            .and_then(|s| {
                let (w, h) = s.split_once('x')?;
                Some([w.trim().parse().ok()?, h.trim().parse().ok()?])
            })
            .unwrap_or([1500.0, 950.0]);
        Some(ShotPlan {
            path,
            tab: var("QB_TAB"),
            preset: var("QB_PRESET").and_then(|s| s.parse().ok()).unwrap_or(0),
            t: var("QB_T").and_then(|s| s.parse().ok()),
            actions,
            present: var("QB_PRESENT").is_some_and(|v| v != "0"),
            light: var("QB_LIGHT").is_some_and(|v| v != "0"),
            seed: var("QB_SEED").and_then(|s| s.parse().ok()).unwrap_or(1),
            size,
            wait: var("QB_WAIT").and_then(|s| s.parse().ok()).unwrap_or(0.5),
        })
    }
}

/// Progress of a running shot.
pub(crate) struct ShotRun {
    pub plan: ShotPlan,
    pub set_up: bool,
    pub frames: u32,
    pub ready_at: Option<crate::Instant>,
    pub requested: bool,
}

impl ShotRun {
    pub fn new(plan: ShotPlan) -> Self {
        ShotRun {
            plan,
            set_up: false,
            frames: 0,
            ready_at: None,
            requested: false,
        }
    }
}

/// Writes an egui screenshot as PNG.
pub(crate) fn save_png(image: &eframe::egui::ColorImage, path: &std::path::Path) -> Result<(), String> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let bytes: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_array()).collect();
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), image.size[0] as u32, image.size[1] as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .and_then(|mut w| w.write_image_data(&bytes))
        .map_err(|e| e.to_string())
}
