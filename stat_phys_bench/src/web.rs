//! Browser helpers: the part of the page address after '#' names the bench.

use wasm_bindgen::prelude::*;

/// the part of the page address after '#', if any
pub fn url_hash() -> Option<String> {
    let hash = web_sys::window()?.location().hash().ok()?;
    let hash = hash.trim_start_matches('#');
    (!hash.is_empty()).then(|| hash.to_string())
}

/// Sets the part after '#' without reloading the page or adding to its history.
pub fn set_url_hash(hash: &str) {
    if let Some(history) = web_sys::window().and_then(|w| w.history().ok()) {
        let _ = history.replace_state_with_url(&JsValue::NULL, "", Some(&format!("#{hash}")));
    }
}
