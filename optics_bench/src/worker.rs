//! A background thread that works on the newest request only.

use std::sync::mpsc;

pub struct Worker<Q, R> {
    to_worker: mpsc::Sender<Q>,
    from_worker: mpsc::Receiver<R>,
    busy: bool,
    queued: Option<Q>,
    last_sent: Option<Q>,
}

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
