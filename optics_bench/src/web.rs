//! Browser helpers for the web version: local storage, downloading files and
//! opening files (with the file dialog or by dropping them onto the page).

use std::cell::RefCell;
use std::rc::Rc;

use eframe::egui;
use wasm_bindgen::JsCast as _;
use wasm_bindgen::prelude::*;

/// A file read by the browser: (file name, contents or error).
pub type ReadFile = (String, Result<String, String>);

/// Text files that are read asynchronously (opened or dropped), picked up by
/// the app on the next frame.
#[derive(Clone, Default)]
pub struct Inbox(Rc<RefCell<Vec<ReadFile>>>);

impl Inbox {
    pub fn take(&self) -> Vec<ReadFile> {
        std::mem::take(&mut *self.0.borrow_mut())
    }

    fn push(&self, ctx: &egui::Context, name: String, text: Result<String, String>) {
        self.0.borrow_mut().push((name, text));
        ctx.request_repaint();
    }
}

/// A readable message for a JavaScript exception.
pub fn js_err(e: JsValue) -> String {
    if let Some(err) = e.dyn_ref::<js_sys::Error>() {
        return String::from(err.message());
    }
    e.as_string().unwrap_or_else(|| format!("{e:?}"))
}

fn document() -> Result<web_sys::Document, String> {
    web_sys::window().and_then(|w| w.document()).ok_or_else(|| "no document".into())
}

pub fn local_storage() -> Result<web_sys::Storage, String> {
    let window = web_sys::window().ok_or("no window")?;
    window.local_storage().map_err(js_err)?.ok_or_else(|| "this browser does not allow local storage".into())
}

/// Saves `text` as a file with the browser's download.
pub fn download(file_name: &str, text: &str) -> Result<(), String> {
    let document = document()?;
    let parts = js_sys::Array::of1(&JsValue::from_str(text));
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("application/json");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options).map_err(js_err)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob).map_err(js_err)?;
    let a: web_sys::HtmlAnchorElement =
        document.create_element("a").map_err(js_err)?.dyn_into().map_err(|_| "no anchor element")?;
    a.set_href(&url);
    a.set_download(file_name);
    a.style().set_property("display", "none").map_err(js_err)?;
    let body = document.body().ok_or("no body")?;
    body.append_child(&a).map_err(js_err)?;
    a.click();
    a.remove();
    // the download may start asynchronously: free the blob a bit later
    let revoke = Closure::once_into_js(move || {
        let _ = web_sys::Url::revoke_object_url(&url);
    });
    web_sys::window()
        .ok_or("no window")?
        .set_timeout_with_callback_and_timeout_and_arguments_0(revoke.unchecked_ref(), 30_000)
        .map_err(js_err)?;
    Ok(())
}

/// Shows the browser's file dialog; the chosen file arrives in `inbox`.
///
/// Browsers only open the dialog shortly after a click or key press, so call
/// this when a button was clicked.
pub fn open_file(ctx: &egui::Context, inbox: &Inbox, accept: &str) -> Result<(), String> {
    const ID: &str = "optics_file_input";
    let document = document()?;
    // one hidden input, kept in the page (Safari wants it attached)
    let input: web_sys::HtmlInputElement = match document.get_element_by_id(ID) {
        Some(el) => el.dyn_into().map_err(|_| "not an input element")?,
        None => {
            let input: web_sys::HtmlInputElement =
                document.create_element("input").map_err(js_err)?.dyn_into().map_err(|_| "no input element")?;
            input.set_id(ID);
            input.set_type("file");
            input.style().set_property("display", "none").map_err(js_err)?;
            document.body().ok_or("no body")?.append_child(&input).map_err(js_err)?;
            input
        }
    };
    input.set_accept(accept);
    // so that choosing the same file again still fires "change"
    input.set_value("");
    let (ctx, inbox, source) = (ctx.clone(), inbox.clone(), input.clone());
    let on_change = Closure::<dyn FnMut()>::new(move || {
        if let Some(file) = source.files().and_then(|files| files.get(0)) {
            read_file(&ctx, &inbox, file);
        }
    });
    input.set_onchange(Some(on_change.as_ref().unchecked_ref()));
    // replaced by the next call; a few bytes per dialog
    on_change.forget();
    input.click();
    Ok(())
}

/// Reads a text file into `inbox`.
pub fn read_file(ctx: &egui::Context, inbox: &Inbox, file: web_sys::File) {
    let (ctx, inbox) = (ctx.clone(), inbox.clone());
    wasm_bindgen_futures::spawn_local(async move {
        let text = wasm_bindgen_futures::JsFuture::from(file.text())
            .await
            .map_err(js_err)
            .and_then(|t| t.as_string().ok_or_else(|| "not a text file".into()));
        inbox.push(&ctx, file.name(), text);
    });
}

/// Reads a file dropped onto the page into `inbox`.
pub fn read_dropped(ctx: &egui::Context, inbox: &Inbox, file: egui::DroppedFileHandle) {
    let (ctx, inbox) = (ctx.clone(), inbox.clone());
    wasm_bindgen_futures::spawn_local(async move {
        let name = file.path().to_string_lossy().into_owned();
        let text = file.bytes_async().await.and_then(|b| String::from_utf8(b).map_err(|e| e.to_string()));
        inbox.push(&ctx, name, text);
    });
}
