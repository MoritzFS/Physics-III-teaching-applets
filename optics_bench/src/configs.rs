//! Saving and loading configurations as JSON files.
//!
//! The files live in a folder "Optics Bench configs" next to the app, so they
//! can be shared by copying them (or dropped onto the window to open them).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::fourier::FourierParams;
use crate::fourier_ui::Mode;
use crate::scene::Scene;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedView {
    pub top_center: [f32; 2],
    pub top_zoom: f32,
    /// target xyz, yaw, pitch, distance
    pub orbit: [f32; 6],
    pub bench_3d: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfigFile {
    pub name: String,
    pub scene: Scene,
    #[serde(default)]
    pub view: Option<SavedView>,
    /// which bench was shown when the configuration was saved
    #[serde(default)]
    pub mode: Mode,
    /// the 4f bench
    #[serde(default)]
    pub fourier: Option<FourierParams>,
    #[serde(default)]
    pub fourier_notes: String,
}

pub struct Entry {
    pub name: String,
    pub path: PathBuf,
}

/// "Optics Bench configs" next to the .app bundle (or next to the source folder
/// when started with `cargo run`).
pub fn dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        for a in exe.ancestors() {
            if a.extension().is_some_and(|e| e == "app") {
                if let Some(parent) = a.parent() {
                    return parent.join("Optics Bench configs");
                }
            }
        }
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap_or(manifest).join("Optics Bench configs")
}

fn file_name(name: &str) -> String {
    let clean: String = name
        .trim()
        .chars()
        .map(|c| if c.is_alphanumeric() || " -_().,".contains(c) { c } else { '_' })
        .collect();
    if clean.is_empty() { "untitled".into() } else { clean }
}

pub fn path_for(name: &str) -> PathBuf {
    dir().join(format!("{}.json", file_name(name)))
}

pub fn list() -> Vec<Entry> {
    let mut entries: Vec<Entry> = std::fs::read_dir(dir())
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .map(|p| Entry { name: p.file_stem().unwrap_or_default().to_string_lossy().into_owned(), path: p })
        .collect();
    entries.sort_by_key(|e| e.name.to_lowercase());
    entries
}

pub fn save(cfg: &ConfigFile) -> Result<PathBuf, String> {
    let path = path_for(&cfg.name);
    std::fs::create_dir_all(dir()).map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| e.to_string())?;
    Ok(path)
}

pub fn load(path: &Path) -> Result<ConfigFile, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn delete(path: &Path) -> Result<(), String> {
    std::fs::remove_file(path).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Preset;

    #[test]
    fn every_preset_survives_a_json_round_trip() {
        for p in Preset::ALL {
            let cfg = ConfigFile {
                name: p.label().into(),
                scene: Scene::preset(p),
                view: Some(SavedView { top_center: [1.0, 2.0], top_zoom: 50.0, orbit: [0.0; 6], bench_3d: true }),
                mode: Mode::Fourier,
                fourier: Some(FourierParams::default()),
                fourier_notes: "x".into(),
            };
            let json = serde_json::to_string_pretty(&cfg).unwrap();
            let back: ConfigFile = serde_json::from_str(&json).unwrap();
            assert_eq!(back.scene, cfg.scene, "{}", p.label());
        }
    }

    #[test]
    fn save_list_load_delete() {
        let cfg = ConfigFile {
            name: "zz test config".into(),
            scene: Scene::preset(Preset::Spectrometer),
            view: None,
            mode: Mode::Ray,
            fourier: None,
            fourier_notes: String::new(),
        };
        let path = save(&cfg).unwrap();
        assert!(list().iter().any(|e| e.name == "zz test config"));
        assert_eq!(load(&path).unwrap().scene, cfg.scene);
        delete(&path).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn older_files_without_notes_or_view_still_load() {
        let mut v = serde_json::to_value(ConfigFile {
            name: "x".into(),
            scene: Scene::empty(),
            view: None,
            mode: Mode::Ray,
            fourier: None,
            fourier_notes: String::new(),
        })
        .unwrap();
        for key in ["view", "mode", "fourier", "fourier_notes"] {
            v.as_object_mut().unwrap().remove(key);
        }
        v["scene"].as_object_mut().unwrap().remove("notes");
        let back: ConfigFile = serde_json::from_value(v).unwrap();
        assert!(back.scene.notes.is_empty());
    }
}
