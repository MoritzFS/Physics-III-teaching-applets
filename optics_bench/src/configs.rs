//! Saving and loading configurations as JSON files.
//!
//! On the desktop the files live in a folder "Optics Bench configs" next to
//! the app, so they can be shared by copying them (or dropped onto the window
//! to open them). In the browser they are kept in the browser's local storage
//! and can be downloaded and opened as .json files.

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
    pub key: Key,
}

/// Where a saved configuration lives: its file on the desktop, its local
/// storage key in the browser.
pub use store::Key;
#[cfg(not(target_arch = "wasm32"))]
pub use store::dir;
pub use store::{delete, exists, list, load, location, save, saved_where};

/// File name for a configuration name (also used for downloads).
pub fn file_name(name: &str) -> String {
    let clean: String = name
        .trim()
        .chars()
        .map(|c| if c.is_alphanumeric() || " -_().,".contains(c) { c } else { '_' })
        .collect();
    format!("{}.json", if clean.is_empty() { "untitled" } else { &clean })
}

pub fn to_json(cfg: &ConfigFile) -> Result<String, String> {
    serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())
}

/// `source` names the file in error messages.
pub fn from_json(text: &str, source: &str) -> Result<ConfigFile, String> {
    serde_json::from_str(text).map_err(|e| format!("{source}: {e}"))
}

/// Desktop: files in a folder next to the app.
#[cfg(not(target_arch = "wasm32"))]
mod store {
    use std::path::{Path, PathBuf};

    use super::{file_name, from_json, to_json, ConfigFile, Entry};

    pub type Key = PathBuf;

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

    fn path_for(name: &str) -> PathBuf {
        dir().join(file_name(name))
    }

    pub fn exists(name: &str) -> bool {
        path_for(name).exists()
    }

    /// shown in the configurations window
    pub fn location() -> String {
        dir().display().to_string()
    }

    /// shown after saving
    pub fn saved_where(key: &Key) -> String {
        key.file_name().unwrap_or_default().to_string_lossy().into_owned()
    }

    pub fn list() -> Vec<Entry> {
        let mut entries: Vec<Entry> = std::fs::read_dir(dir())
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .map(|p| Entry { name: p.file_stem().unwrap_or_default().to_string_lossy().into_owned(), key: p })
            .collect();
        entries.sort_by_key(|e| e.name.to_lowercase());
        entries
    }

    pub fn save(cfg: &ConfigFile) -> Result<Key, String> {
        let path = path_for(&cfg.name);
        std::fs::create_dir_all(dir()).map_err(|e| e.to_string())?;
        std::fs::write(&path, to_json(cfg)?).map_err(|e| e.to_string())?;
        Ok(path)
    }

    /// Opens a configuration file (saved, dropped or given on the command line).
    pub fn load(path: &Path) -> Result<ConfigFile, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        from_json(&text, &path.display().to_string())
    }

    pub fn delete(path: &Path) -> Result<(), String> {
        std::fs::remove_file(path).map_err(|e| e.to_string())
    }
}

/// Browser: the local storage of this site, one key per configuration.
#[cfg(target_arch = "wasm32")]
mod store {
    use super::{from_json, to_json, ConfigFile, Entry};
    use crate::web::{js_err, local_storage};

    pub type Key = String;

    const PREFIX: &str = "optics_bench/config/";

    fn key_for(name: &str) -> Key {
        format!("{PREFIX}{}", name.trim())
    }

    pub fn exists(name: &str) -> bool {
        local_storage().ok().and_then(|s| s.get_item(&key_for(name)).ok().flatten()).is_some()
    }

    /// shown in the configurations window
    pub fn location() -> String {
        "Saved in this browser only. Use Download to keep a copy or to share it.".into()
    }

    /// shown after saving
    pub fn saved_where(_key: &Key) -> String {
        "in this browser".into()
    }

    pub fn list() -> Vec<Entry> {
        let Ok(storage) = local_storage() else { return vec![] };
        let n = storage.length().unwrap_or(0);
        let mut entries: Vec<Entry> = (0..n)
            .filter_map(|i| storage.key(i).ok().flatten())
            .filter_map(|key| Some(Entry { name: key.strip_prefix(PREFIX)?.to_string(), key: key.clone() }))
            .collect();
        entries.sort_by_key(|e| e.name.to_lowercase());
        entries
    }

    pub fn save(cfg: &ConfigFile) -> Result<Key, String> {
        let key = key_for(&cfg.name);
        local_storage()?.set_item(&key, &to_json(cfg)?).map_err(js_err)?;
        Ok(key)
    }

    pub fn load(key: &str) -> Result<ConfigFile, String> {
        let name = key.strip_prefix(PREFIX).unwrap_or(key);
        let text = local_storage()?.get_item(key).map_err(js_err)?.ok_or_else(|| format!("'{name}' is gone"))?;
        from_json(&text, name)
    }

    pub fn delete(key: &str) -> Result<(), String> {
        local_storage()?.remove_item(key).map_err(js_err)
    }
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
        let key = save(&cfg).unwrap();
        assert!(exists("zz test config"));
        assert!(list().iter().any(|e| e.name == "zz test config"));
        assert_eq!(load(&key).unwrap().scene, cfg.scene);
        delete(&key).unwrap();
        assert!(!exists("zz test config"));
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
