//! Plays mono sounds on the default output device. The device is only opened
//! when something is played for the first time.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

static NEXT_CLIP: AtomicU64 = AtomicU64::new(1);

/// a new id to recognise a clip by
pub fn clip_id() -> u64 {
    NEXT_CLIP.fetch_add(1, Ordering::Relaxed)
}

#[derive(Default)]
struct State {
    clip: Option<Arc<Vec<f32>>>,
    /// sample rate of the clip and of the device
    rate: f64,
    device_rate: f64,
    /// position in samples of the clip
    pos: f64,
    id: u64,
}

#[derive(Default)]
pub struct Audio {
    stream: Option<cpal::Stream>,
    state: Arc<Mutex<State>>,
    pub error: Option<String>,
}

impl Audio {
    fn open(&mut self) -> bool {
        if self.stream.is_some() {
            return true;
        }
        match open_stream(self.state.clone()) {
            Ok(s) => {
                self.stream = Some(s);
                self.error = None;
                true
            }
            Err(e) => {
                self.error = Some(format!("no sound: {e}"));
                false
            }
        }
    }

    /// plays `clip` (mono, `rate` samples per second) from `start` seconds on
    pub fn play(&mut self, clip: Arc<Vec<f32>>, rate: u32, id: u64, start: f32) {
        if !self.open() {
            return;
        }
        let mut s = self.state.lock().unwrap();
        s.pos = (start.max(0.0) as f64 * rate as f64).min(clip.len() as f64);
        s.clip = Some(clip);
        s.rate = rate as f64;
        s.id = id;
    }

    pub fn stop(&mut self) {
        self.state.lock().unwrap().clip = None;
    }

    /// the clip that is playing and the position in it (s)
    pub fn playing(&self) -> Option<(u64, f32)> {
        let s = self.state.lock().unwrap();
        s.clip.as_ref().map(|_| (s.id, (s.pos / s.rate) as f32))
    }
}

fn open_stream(state: Arc<Mutex<State>>) -> Result<cpal::Stream, String> {
    let device = cpal::default_host().default_output_device().ok_or("no output device")?;
    let cfg = device.default_output_config().map_err(|e| e.to_string())?;
    let channels = cfg.channels() as usize;
    state.lock().unwrap().device_rate = cfg.sample_rate().0 as f64;
    let config = cfg.config();
    let err = |e: cpal::StreamError| eprintln!("audio: {e}");
    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => {
            device.build_output_stream(&config, move |d: &mut [f32], _: &_| fill(d, channels, &state, |v| v), err, None)
        }
        cpal::SampleFormat::I16 => device.build_output_stream(
            &config,
            move |d: &mut [i16], _: &_| fill(d, channels, &state, |v| (v * 32767.0) as i16),
            err,
            None,
        ),
        f => return Err(format!("unsupported sample format {f:?}")),
    }
    .map_err(|e| e.to_string())?;
    stream.play().map_err(|e| e.to_string())?;
    Ok(stream)
}

fn fill<T: Copy>(data: &mut [T], channels: usize, state: &Mutex<State>, conv: impl Fn(f32) -> T) {
    let Ok(mut s) = state.try_lock() else {
        data.fill(conv(0.0));
        return;
    };
    let step = s.rate / s.device_rate.max(1.0);
    for frame in data.chunks_mut(channels.max(1)) {
        let v = match s.clip.as_ref() {
            Some(c) if (s.pos as usize) + 1 < c.len() => {
                let i = s.pos as usize;
                let a = (s.pos - i as f64) as f32;
                c[i] * (1.0 - a) + c[i + 1] * a
            }
            _ => 0.0,
        };
        frame.fill(conv(v));
        if s.clip.is_some() {
            s.pos += step;
            if s.clip.as_ref().is_some_and(|c| s.pos as usize + 1 >= c.len()) {
                s.clip = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn plays_a_quiet_clip() {
        let mut a = Audio::default();
        let clip = Arc::new(vec![0.0f32; 12000]);
        a.play(clip, 48000, 7, 0.0);
        assert!(a.error.is_none(), "{:?}", a.error);
        std::thread::sleep(std::time::Duration::from_millis(120));
        let (id, pos) = a.playing().expect("still playing");
        assert_eq!(id, 7);
        assert!(pos > 0.02 && pos < 0.25, "position {pos}");
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(a.playing().is_none(), "should have finished");
    }
}
