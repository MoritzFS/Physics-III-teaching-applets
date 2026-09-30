//! Works on the newest request only: on a background thread on the desktop,
//! and in place (in `poll`, at most once per frame) in the browser, which has
//! no threads.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;

#[cfg(not(target_arch = "wasm32"))]
pub struct Worker<Q, R> {
    to_worker: mpsc::Sender<Q>,
    from_worker: mpsc::Receiver<R>,
    busy: bool,
    queued: Option<Q>,
    last_sent: Option<Q>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<Q: Clone + PartialEq + Send + 'static, R: Send + 'static> Worker<Q, R> {
    pub fn new(name: &str, mut work: impl FnMut(Q) -> R + Send + 'static, repaint: impl Fn() + Send + 'static) -> Self {
        let (to_worker, rx) = mpsc::channel::<Q>();
        let (tx, from_worker) = mpsc::channel();
        std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                while let Ok(mut q) = rx.recv() {
                    while let Ok(newer) = rx.try_recv() {
                        q = newer;
                    }
                    if tx.send(work(q)).is_err() {
                        break;
                    }
                    repaint();
                }
            })
            .expect("worker thread");
        Worker { to_worker, from_worker, busy: false, queued: None, last_sent: None }
    }

    /// ask for a new computation unless this one was already asked for
    pub fn request(&mut self, q: &Q) {
        if self.last_sent.as_ref() == Some(q) || self.queued.as_ref() == Some(q) {
            return;
        }
        if self.busy {
            self.queued = Some(q.clone());
        } else {
            let _ = self.to_worker.send(q.clone());
            self.last_sent = Some(q.clone());
            self.busy = true;
        }
    }

    pub fn poll(&mut self) -> Option<R> {
        let r = self.from_worker.try_recv().ok()?;
        self.busy = false;
        if let Some(q) = self.queued.take() {
            let _ = self.to_worker.send(q.clone());
            self.last_sent = Some(q);
            self.busy = true;
        }
        Some(r)
    }

    pub fn busy(&self) -> bool {
        self.busy
    }
}

/// The browser has no threads: computes in `poll`, at most once per frame,
/// always for the newest request. A computation blocks the page, so after a
/// slow one (thunder: about 1 s) the next waits until the request has stopped
/// changing, e.g. until a slider is let go.
#[cfg(target_arch = "wasm32")]
pub struct Worker<Q, R> {
    work: Box<dyn FnMut(Q) -> R>,
    repaint: Box<dyn Fn()>,
    queued: Option<(Q, web_time::Instant)>,
    last_done: Option<Q>,
    last_seconds: f32,
}

#[cfg(target_arch = "wasm32")]
impl<Q: Clone + PartialEq + Send + 'static, R: Send + 'static> Worker<Q, R> {
    pub fn new(_name: &str, work: impl FnMut(Q) -> R + Send + 'static, repaint: impl Fn() + Send + 'static) -> Self {
        Worker { work: Box::new(work), repaint: Box::new(repaint), queued: None, last_done: None, last_seconds: 0.0 }
    }

    /// ask for a new computation unless this one was already done
    pub fn request(&mut self, q: &Q) {
        if self.last_done.as_ref() == Some(q) {
            self.queued = None;
        } else if self.queued.as_ref().is_none_or(|(old, _)| old != q) {
            self.queued = Some((q.clone(), web_time::Instant::now()));
            // the result is computed in a later frame's poll
            (self.repaint)();
        }
    }

    pub fn poll(&mut self) -> Option<R> {
        let (_, since) = self.queued.as_ref()?;
        if self.last_seconds > 0.1 && since.elapsed().as_secs_f32() < 0.2 {
            (self.repaint)();
            return None;
        }
        let (q, _) = self.queued.take()?;
        let t0 = web_time::Instant::now();
        let r = (self.work)(q.clone());
        self.last_seconds = t0.elapsed().as_secs_f32();
        self.last_done = Some(q);
        Some(r)
    }

    pub fn busy(&self) -> bool {
        self.queued.is_some()
    }
}
