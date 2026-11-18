//! Browser start-up and helpers (wasm32 only).
//!
//! `index.html` provides a canvas `#qb_canvas` and the messages `#loading`,
//! `#no_gpu` (with `#no_gpu_detail`) and `#crashed` (with `#crashed_detail`).
//! wgpu prefers WebGPU and falls back to WebGL2 (eframe's default), so all
//! custom GPU code must stay within WebGL2 limits (see `qb_core::gpu`).
//! Add `?webgl` to the URL to force WebGL2, e.g. `http://127.0.0.1:8080/?webgl#hydrogen`.

use wasm_bindgen::JsCast as _;

pub(crate) const CANVAS_ID: &str = "qb_canvas";

fn document() -> Option<web_sys::Document> {
    web_sys::window().and_then(|w| w.document())
}

/// The URL fragment without '#', e.g. "wave1d/3".
pub(crate) fn location_hash() -> String {
    web_sys::window()
        .and_then(|w| w.location().hash().ok())
        .map(|h| h.trim_start_matches('#').to_string())
        .unwrap_or_default()
}

/// Replaces the URL fragment without adding a history entry.
pub(crate) fn set_location_hash(hash: &str) {
    let Some(window) = web_sys::window() else { return };
    if let Ok(history) = window.history() {
        let _ = history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&format!("#{hash}")));
    }
}

/// Starts the app in the page's canvas.
pub(crate) fn start(title: String, creator: eframe::AppCreator<'static>) {
    eframe::WebLogger::init(log::LevelFilter::Info).ok();
    // eframe's own panic hook (console) calls this one afterwards
    std::panic::set_hook(Box::new(|info| show_crash(&info.to_string())));
    let mut web_options = eframe::WebOptions::default();
    // wgpu prefers WebGPU and falls back to WebGL2; `?webgl` in the URL forces WebGL2 (for testing)
    let force_gl = web_sys::window()
        .and_then(|w| w.location().search().ok())
        .is_some_and(|s| s.contains("webgl"));
    if force_gl && let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut web_options.wgpu_options.wgpu_setup {
        setup.instance_descriptor.backends = eframe::egui_wgpu::wgpu::Backends::GL;
        log::info!("?webgl: using the WebGL2 backend");
    }
    wasm_bindgen_futures::spawn_local(async move {
        let Some(document) = document() else { return };
        if !title.is_empty() {
            document.set_title(&title);
        }
        let Some(canvas) = document.get_element_by_id(CANVAS_ID) else {
            log::warn!("no #{CANVAS_ID} in the page: not starting");
            return;
        };
        let Ok(canvas) = canvas.dyn_into::<web_sys::HtmlCanvasElement>() else {
            log::error!("#{CANVAS_ID} is not a canvas");
            return;
        };
        let result = eframe::WebRunner::new().start(canvas, web_options, creator).await;
        if let Some(loading) = document.get_element_by_id("loading") {
            loading.remove();
        }
        if let Err(e) = result {
            let detail = e.as_string().unwrap_or_else(|| format!("{e:?}"));
            log::error!("could not start: {detail}");
            if let Some(canvas) = document.get_element_by_id(CANVAS_ID) {
                canvas.remove();
            }
            if let Some(msg) = document.get_element_by_id("no_gpu") {
                let _ = msg.remove_attribute("hidden");
            }
            if let Some(el) = document.get_element_by_id("no_gpu_detail") {
                el.set_text_content(Some(&detail));
            }
        }
    });
}

/// A panic ends the app: say so in the page instead of leaving it frozen.
fn show_crash(message: &str) {
    let Some(document) = document() else { return };
    if let Some(loading) = document.get_element_by_id("loading") {
        loading.remove();
    }
    if let Some(canvas) = document.get_element_by_id(CANVAS_ID) {
        let _ = canvas.set_attribute("data-crashed", "");
    }
    if let Some(el) = document.get_element_by_id("crashed") {
        let _ = el.remove_attribute("hidden");
    }
    if let Some(el) = document.get_element_by_id("crashed_detail") {
        el.set_text_content(Some(&format!("{message} (more in the browser's developer console)")));
    }
}
