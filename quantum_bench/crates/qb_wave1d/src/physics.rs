//! Closed-form results of the exercises (ħ = m = 1), used by the scenes for
//! readouts and reference curves, and checked against the numerics in the tests.

use std::f64::consts::{PI, TAU};

use qb_core::num::{C64, Momentum, c64};

// ------------------------------------------------------------ PS09 37

/// Φ(p) of the truncated plane wave Ψ = e^(i2πx/λ)/√(2nλ) on (−nλ, nλ)
/// (PS09 37 a): Φ(p) = √(λ/(nπ)) sin(npλ)/(pλ − 2π), real (box centred on 0).
pub fn truncated_phi(p: f64, n: f64, lambda: f64) -> f64 {
    let d = p * lambda - TAU;
    if d.abs() < 1e-9 {
        // limit p → 2π/λ: sin(npλ)/(pλ − 2π) → n
        return (lambda / (n * PI)).sqrt() * n;
    }
    (lambda / (n * PI)).sqrt() * (n * p * lambda).sin() / d
}

/// Widths between the zeros either side of the main peak (PS09 37 b):
/// wₓ = 2nλ, wₚ = 2πħ/(nλ) = h/(nλ).
pub fn truncated_widths(n: f64, lambda: f64) -> (f64, f64) {
    (2.0 * n * lambda, TAU / (n * lambda))
}

/// Positions of the zeros (or of the first minima that drop below 5 % of the
/// peak) on either side of the main peak of |amp|² on a uniform grid.
/// A jump to zero (a sharp edge) counts as a zero at the midpoint of the jump;
/// at a minimum, the complex amplitude (which changes sign through a simple
/// zero) is fitted by a parabola through three points and its root is taken.
/// Returns None on a side where the density only fades away (e.g. a Gaussian),
/// and the index of the main peak.
pub fn zeros_around_peak(xs: &[f64], amp: &[C64]) -> (Option<f64>, Option<f64>, usize) {
    let n = amp.len();
    if n < 3 {
        return (None, None, 0);
    }
    let d: Vec<f64> = amp.iter().map(|z| z.norm_sqr()).collect();
    let i0 = d.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0);
    let peak = d[i0];
    if peak <= 0.0 {
        return (None, None, i0);
    }
    let tiny = 1e-7 * peak;
    let h = xs[1] - xs[0];
    let find = |dir: isize| -> Option<f64> {
        let mut j = i0 as isize;
        loop {
            let next = j + dir;
            if next < 0 || next >= n as isize {
                return None;
            }
            let (dj, dn) = (d[j as usize], d[next as usize]);
            if dn <= tiny {
                // a jump to zero from a clearly non-zero value is an edge; a slow fade is not a zero
                return (dj > 1e-3 * peak).then(|| 0.5 * (xs[j as usize] + xs[next as usize]));
            }
            if dn > dj && dj < 0.05 * peak {
                let ju = j as usize;
                if ju == 0 || ju + 1 >= n {
                    return Some(xs[ju]);
                }
                // f(u) = a + b·u + c·u² through u = −1, 0, 1 (units of h), root nearest to 0
                let (fm, f0, fp) = (amp[ju - 1], amp[ju], amp[ju + 1]);
                let (a, b, c) = (f0, (fp - fm) * 0.5, (fp + fm) * 0.5 - f0);
                let mut u = C64::new(0.0, 0.0);
                for _ in 0..8 {
                    let df = b + c * u * 2.0;
                    if df.norm() < 1e-300 {
                        break;
                    }
                    u -= (a + b * u + c * u * u) / df;
                }
                return Some(xs[ju] + u.re.clamp(-1.0, 1.0) * h);
            }
            j = next;
        }
    };
    (find(-1), find(1), i0)
}

// ------------------------------------------------------------ PS10 39

/// Energies above a rectangular barrier where T = 1 (PS10 39 d): k₂a = nπ,
/// i.e. E = V + (nπ/a)²/2, n = 1, 2, … up to `e_max`.
pub fn barrier_resonances(v0: f64, a: f64, e_max: f64) -> Vec<f64> {
    (1..200)
        .map(|n| v0 + 0.5 * (n as f64 * PI / a).powi(2))
        .take_while(|&e| e <= e_max)
        .collect()
}

/// Probability that a packet with momentum amplitude Φ(p) ends up transmitted:
/// ∫_{p>0} |Φ(p)|² T(p²/2) dp / ∫|Φ|² dp. Components with p < 0 move away
/// from the barrier and count as reflected.
pub fn packet_transmission(mom: &Momentum, t_of_e: impl Fn(f64) -> f64) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (c, &p) in mom.phi.iter().zip(&mom.p) {
        let w = c.norm_sqr();
        den += w;
        if p > 0.0 {
            num += w * t_of_e(0.5 * p * p);
        }
    }
    if den > 0.0 { num / den } else { 0.0 }
}

/// Energy distribution of a free packet, P(E) = (|Φ(√2E)|² + |Φ(−√2E)|²)/√(2E),
/// sampled at `energies` (normalised so that ∫P dE = ∫|Φ|² dp).
pub fn energy_density(mom: &Momentum, energies: &[f64]) -> Vec<f64> {
    let dens = mom.density();
    let interp = |p: f64| -> f64 {
        let f = (p - mom.p[0]) / mom.dp;
        if f < 0.0 || f >= (mom.p.len() - 1) as f64 {
            return 0.0;
        }
        let i = f.floor() as usize;
        let t = f - i as f64;
        dens[i] * (1.0 - t) + dens[i + 1] * t
    };
    energies
        .iter()
        .map(|&e| {
            if e <= 0.0 {
                return 0.0;
            }
            let p = (2.0 * e).sqrt();
            (interp(p) + interp(-p)) / p
        })
        .collect()
}

// ------------------------------------------------------------ PS10 40

/// Poisson weight |c_n|² = e^(−μ) μⁿ/n! of a coherent state with μ = |α|².
pub fn poisson(n: usize, mean: f64) -> f64 {
    let mut lp = -mean;
    for k in 1..=n {
        lp += mean.max(1e-300).ln() - (k as f64).ln();
    }
    if mean == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    lp.exp()
}

/// σₓσₚ = (2n + 1)/2 of the oscillator number state ψₙ (ħ = 1).
pub fn number_state_product(n: usize) -> f64 {
    n as f64 + 0.5
}

/// ⟨x⟩, ⟨p⟩ of the coherent state |α e^(−iωt)⟩ (ħ = m = 1): √(2/ω) Re α(t), √(2ω) Im α(t).
pub fn coherent_mean(alpha: C64, omega: f64, t: f64) -> (f64, f64) {
    let a = alpha * C64::from_polar(1.0, -omega * t);
    ((2.0 / omega).sqrt() * a.re, (2.0 * omega).sqrt() * a.im)
}

// ------------------------------------------------------------ PS12 49

/// Exact levels of H = p²/2 + ξ²/2 − λξ (units ħω): Eₙ = n + ½ − λ²/2.
pub fn shifted_energy(n: usize, lambda: f64) -> f64 {
    n as f64 + 0.5 - 0.5 * lambda * lambda
}

/// First-order perturbation theory: Eₙ⁽¹⁾ = n + ½ (⟨n|ξ|n⟩ = 0).
pub fn first_order_energy(n: usize, _lambda: f64) -> f64 {
    n as f64 + 0.5
}

/// Up to second order: Eₙ = n + ½ + λ²[(n+1)/2/(−1) + n/2/(+1)] = n + ½ − λ²/2 (equal to the exact value).
pub fn second_order_energy(n: usize, lambda: f64) -> f64 {
    let l2 = lambda * lambda;
    n as f64 + 0.5 + l2 * (-(n as f64 + 1.0) / 2.0 + n as f64 / 2.0)
}

/// Coefficients (in the unperturbed basis, length `count`) of the first-order
/// state ψₙ⁽¹⁾ = ψₙ + λ√((n+1)/2) ψₙ₊₁ − λ√(n/2) ψₙ₋₁ (PS12 49 c), not normalised.
pub fn first_order_state(n: usize, lambda: f64, count: usize) -> Vec<C64> {
    let mut c = vec![c64(0.0, 0.0); count];
    if n < count {
        c[n] = c64(1.0, 0.0);
    }
    if n + 1 < count {
        c[n + 1] = c64(lambda * ((n as f64 + 1.0) / 2.0).sqrt(), 0.0);
    }
    if n >= 1 && n - 1 < count {
        c[n - 1] = c64(-lambda * (n as f64 / 2.0).sqrt(), 0.0);
    }
    c
}

/// ⟨ξ⟩ in the normalised first-order state: λ/(1 + λ²(2n+1)/2) = λ + O(λ³).
pub fn first_order_mean_xi(n: usize, lambda: f64) -> f64 {
    lambda / (1.0 + 0.5 * lambda * lambda * (2.0 * n as f64 + 1.0))
}

// ------------------------------------------------------------ PS09 36

/// Survival probability P(t) = e^(−Γt) (PS09 36 b), τ = 1/Γ.
pub fn survival(gamma: f64, t: f64) -> f64 {
    (-gamma * t).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use qb_core::num::Wave1D;

    #[test]
    fn truncated_plane_wave_fourier_transform_and_widths() {
        let g = crate::truncated::grid();
        for &(n, lambda) in &[(3.0, 1.0), (5.0, 2.0), (1.0, 1.5), (12.0, 3.7)] {
            let w = Wave1D::plane_wave_truncated(g, 0.0, n * lambda, TAU / lambda);
            let m = w.momentum();
            // the FFT of the sampled box agrees with the analytic Φ(p) near the main peak
            let p0 = TAU / lambda;
            for k in 0..m.p.len() {
                let p = m.p[k];
                if (p - p0).abs() < 3.0 / (n * lambda) {
                    let want = truncated_phi(p, n, lambda);
                    assert!((m.phi[k].norm() - want.abs()).abs() < 0.02 * (n * lambda / PI).sqrt(), "|Φ({p})|");
                }
            }
            // zeros either side of the main peaks
            let (wx, wp) = truncated_widths(n, lambda);
            let (a, b, _) = zeros_around_peak(&w.xs(), &w.psi);
            let num_wx = b.unwrap() - a.unwrap();
            // the sampled box is exactly 2nλ wide on this grid
            assert!((num_wx - wx).abs() < 1e-9, "wₓ {num_wx} vs {wx}");
            assert!((w.norm_sqr() - 1.0).abs() < 1e-12);
            let (a, b, i0) = zeros_around_peak(&m.p, &m.phi);
            let num_wp = b.unwrap() - a.unwrap();
            assert!((num_wp - wp).abs() < 2e-3 * wp, "wₚ {num_wp} vs {wp}");
            assert!((m.p[i0] - p0).abs() <= m.dp, "peak at h/λ");
            // wₓ·wₚ = 4πħ = 2h, independent of n and λ
            assert!((wx * wp - 4.0 * PI).abs() < 1e-12);
        }
        // a Gaussian has no zeros
        let w = Wave1D::gaussian(g, 0.0, 2.0, 1.0);
        let (a, b, _) = zeros_around_peak(&w.xs(), &w.psi);
        assert!(a.is_none() && b.is_none());
    }

    #[test]
    fn poisson_weights_sum_and_mean() {
        let mu = 4.0;
        let w: Vec<f64> = (0..60).map(|n| poisson(n, mu)).collect();
        let s: f64 = w.iter().sum();
        let mean: f64 = w.iter().enumerate().map(|(n, p)| n as f64 * p).sum();
        let var: f64 = w.iter().enumerate().map(|(n, p)| (n as f64 - mean).powi(2) * p).sum();
        assert!((s - 1.0).abs() < 1e-12 && (mean - mu).abs() < 1e-10 && (var - mu).abs() < 1e-9);
        assert_eq!(poisson(0, 0.0), 1.0);
    }

    #[test]
    fn second_order_equals_exact_shift() {
        for n in 0..6 {
            for &l in &[0.0, 0.3, 1.0, 1.7] {
                assert!((second_order_energy(n, l) - shifted_energy(n, l)).abs() < 1e-14);
                assert_eq!(first_order_energy(n, l), n as f64 + 0.5);
            }
        }
    }

    #[test]
    fn barrier_resonance_energies_have_unit_transmission() {
        for &(v, a) in &[(1.0, 3.0), (2.0, 1.5)] {
            for e in barrier_resonances(v, a, 10.0) {
                let t = qb_core::num::rect_barrier_transmission(e, v, a);
                assert!((t - 1.0).abs() < 1e-12, "T({e}) = {t}");
            }
        }
    }
}
