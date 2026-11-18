//! Data parallelism that also works in the browser.
//!
//! On the desktop these use rayon; on the web (no threads) they run
//! sequentially on the current thread. Tab crates should use these instead of
//! rayon directly, so the wasm build keeps compiling.
//!
//! ```
//! use qb_core::par;
//! let squares = par::map_range(1000, |i| (i * i) as f64);
//! let mut image = vec![0u8; 64 * 32];
//! par::chunks_mut(&mut image, 64, |row, px| px.fill(row as u8)); // one row per call
//! # assert_eq!(squares[3], 9.0);
//! ```

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// `items.iter().map(f).collect()`, in parallel on the desktop.
pub fn map<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        items.par_iter().map(f).collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        items.iter().map(f).collect()
    }
}

/// `(0..n).map(f).collect()`, in parallel on the desktop.
pub fn map_range<R, F>(n: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..n).map(f).collect()
    }
}

/// Calls `f(index, &mut item)` for every item, in parallel on the desktop.
pub fn for_each_mut<T, F>(items: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Sync + Send,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        items.par_iter_mut().enumerate().for_each(|(i, x)| f(i, x));
    }
    #[cfg(target_arch = "wasm32")]
    {
        items.iter_mut().enumerate().for_each(|(i, x)| f(i, x));
    }
}

/// Calls `f(chunk_index, chunk)` for consecutive chunks of `chunk_len` items
/// (e.g. the rows of an image), in parallel on the desktop.
pub fn chunks_mut<T, F>(items: &mut [T], chunk_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    let chunk_len = chunk_len.max(1);
    #[cfg(not(target_arch = "wasm32"))]
    {
        items.par_chunks_mut(chunk_len).enumerate().for_each(|(i, c)| f(i, c));
    }
    #[cfg(target_arch = "wasm32")]
    {
        items.chunks_mut(chunk_len).enumerate().for_each(|(i, c)| f(i, c));
    }
}

/// Sum of `f(i)` for i in 0..n, in parallel on the desktop.
pub fn sum_range<F>(n: usize, f: F) -> f64
where
    F: Fn(usize) -> f64 + Sync + Send,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        (0..n).into_par_iter().map(f).sum()
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..n).map(f).sum()
    }
}

/// true when the helpers above really run in parallel (desktop).
pub const PARALLEL: bool = cfg!(not(target_arch = "wasm32"));
