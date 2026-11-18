//! Physics checks of the 1D numerics (run with `cargo test -p qb_core`).

use std::sync::Arc;

use qb_core::QbRng;
use qb_core::num::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn parseval_and_momentum_of_a_gaussian() {
    let g = Grid1D::new(-40.0, 40.0, 2048);
    let w = Wave1D::gaussian(g, 1.5, 0.8, 2.0);
    let m = w.momentum();
    assert!(close(w.norm_sqr(), 1.0, 1e-12));
    assert!(
        close(m.norm_sqr(), w.norm_sqr(), 1e-12),
        "Parseval: {} vs {}",
        m.norm_sqr(),
        w.norm_sqr()
    );
    let mo = w.moments();
    assert!(close(mo.x, 1.5, 1e-9) && close(mo.sigma_x, 0.8, 1e-9));
    assert!(close(mo.p, 2.0, 1e-9), "⟨p⟩ = {}", mo.p);
    assert!(close(mo.sigma_p, 1.0 / (2.0 * 0.8), 1e-9), "σ_p = {}", mo.sigma_p);
    assert!(close(mo.uncertainty_product(), 0.5, 1e-9));
    // Φ(p) = (2σ²/π)^(1/4) exp(−σ²(p−p0)² − i(p−p0)x0) · e^(−i p0 x0) … check modulus and phase at p0 ± 0.5
    let s = 0.8f64;
    for &p in &[2.0, 2.5, 1.5] {
        let k = g.p_index_f(p).round() as usize;
        let pk = m.p[k];
        let expect = (2.0 * s * s / std::f64::consts::PI).powf(0.25) * (-s * s * (pk - 2.0).powi(2)).exp();
        assert!(close(m.phi[k].norm(), expect, 1e-9), "|Φ({pk})|");
        // phase: −(p − p0)·x0 for ψ ∝ g(x − x0) e^(i p0 x) (absolute-x convention)
        let phase = m.phi[k].arg();
        let want = (-(pk - 2.0) * 1.5).rem_euclid(std::f64::consts::TAU);
        let d = (phase - want).rem_euclid(std::f64::consts::TAU);
        assert!(d.min(std::f64::consts::TAU - d) < 1e-8, "arg Φ({pk}) = {phase}, want {want}");
    }
    // round trip
    let back = Wave1D::from_momentum(&m);
    let err: f64 = back.psi.iter().zip(&w.psi).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
    assert!(err < 1e-12, "round trip error {err}");
}

#[test]
fn free_gaussian_spreads_and_conserves_norm() {
    let g = Grid1D::new(-60.0, 60.0, 4096);
    let s0 = 1.0;
    let mut w = Wave1D::gaussian(g, -5.0, s0, 1.0);
    let mut prop = SplitStep::new(g, 0.05);
    let mut t = 0.0;
    for _ in 0..8 {
        prop.advance(&mut w, 1.0);
        t += 1.0;
        let mo = w.moments();
        assert!(close(mo.norm, 1.0, 1e-10), "norm {}", mo.norm);
        let want = free_gaussian_sigma(s0, t);
        assert!(close(mo.sigma_x, want, 1e-6), "σ_x({t}) = {} vs {want}", mo.sigma_x);
        assert!(close(mo.x, -5.0 + t, 1e-6), "⟨x⟩({t}) = {}", mo.x);
        assert!(close(mo.sigma_p, 0.5 / s0, 1e-9));
    }
    // against the analytic ψ(x, t), phase included
    let err = (0..g.n)
        .map(|i| (w.psi[i] - free_gaussian_psi(g.x(i), t, -5.0, s0, 1.0)).norm())
        .fold(0.0, f64::max);
    assert!(err < 1e-6, "ψ(x, t) error {err}");
}

#[test]
fn norm_conserved_with_real_potential() {
    let g = Grid1D::new(-20.0, 20.0, 1024);
    let v = g.sample(|x| 0.5 * x * x + 2.0 * (-(x - 1.0) * (x - 1.0)).exp());
    let mut w = Wave1D::gaussian(g, -3.0, 0.7, 0.5);
    let mut prop = SplitStep::new(g, 0.01).with_potential(&v);
    let e0 = w.energy(&v);
    prop.advance(&mut w, 10.0);
    assert!(close(w.norm_sqr(), 1.0, 1e-10));
    assert!(close(w.energy(&v), e0, 1e-3), "energy drift {} → {}", e0, w.energy(&v));
}

#[test]
fn imaginary_potential_decays_as_exp_minus_gamma_t() {
    // PS09 36: V = V0 − iΓ/2 ⇒ P(t) = e^(−Γt)
    let g = Grid1D::new(-30.0, 30.0, 1024);
    let gamma = 0.3;
    let v: Vec<C64> = (0..g.n).map(|_| c64(0.2, -gamma / 2.0)).collect();
    let mut w = Wave1D::gaussian(g, 0.0, 1.0, 0.0);
    let mut prop = SplitStep::new(g, 0.02);
    prop.set_complex_potential(&v);
    for k in 1..=5 {
        prop.advance(&mut w, 1.0);
        let t = k as f64;
        assert!(close(w.norm_sqr(), (-gamma * t).exp(), 1e-10), "P({t}) = {}", w.norm_sqr());
    }
}

#[test]
fn absorber_removes_outgoing_packet() {
    let g = Grid1D::new(-40.0, 40.0, 2048);
    let mut w = Wave1D::gaussian(g, 0.0, 1.5, 3.0);
    let mut prop = SplitStep::new(g, 0.02).with_absorber(Absorber::new(10.0, 4.0));
    prop.advance(&mut w, 25.0); // the packet has run through the right layer
    assert!(w.norm_sqr() < 1e-3, "left over {}", w.norm_sqr());
}

#[test]
fn harmonic_oscillator_levels() {
    let g = Grid1D::new(-12.0, 12.0, 2400);
    let v = g.sample(|x| 0.5 * x * x);
    let b = EigenBasis::numeric(g, &v, 12);
    for (n, e) in b.energies.iter().enumerate() {
        assert!(close(*e, n as f64 + 0.5, 1e-3), "E_{n} = {e}");
    }
    // orthonormal
    for i in 0..b.len() {
        for j in 0..b.len() {
            let d: f64 = b.states[i].iter().zip(&b.states[j]).map(|(a, c)| a * c).sum::<f64>() * g.dx;
            assert!(close(d, if i == j { 1.0 } else { 0.0 }, 1e-6), "⟨{i}|{j}⟩ = {d}");
        }
    }
    // agrees with the analytic states up to the documented sign (−1)ⁿ
    let a = EigenBasis::harmonic(g, 12, 1.0, 0.0);
    for n in 0..12 {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let err = b.states[n]
            .iter()
            .zip(&a.states[n])
            .map(|(x, y)| (x - sign * y).abs())
            .fold(0.0, f64::max);
        assert!(err < 5e-3, "ψ_{n} differs by {err}");
    }
    // first lobe positive
    for s in &b.states {
        let max = s.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(*s.iter().find(|x| x.abs() > 1e-3 * max).unwrap() > 0.0);
    }
}

#[test]
fn double_well_splitting_and_infinite_well() {
    // close levels stay orthogonal
    let g = Grid1D::new(-6.0, 6.0, 1200);
    let v = g.sample(|x| 8.0 * (x * x - 2.0f64).powi(2) / 4.0);
    let b = EigenBasis::numeric(g, &v, 4);
    let split = b.energies[1] - b.energies[0];
    assert!(split > 0.0 && split < 0.1, "splitting {split}");
    let d: f64 = b.states[0].iter().zip(&b.states[1]).map(|(a, c)| a * c).sum::<f64>() * g.dx;
    assert!(d.abs() < 1e-6, "overlap {d}");
    // a deep square well approaches n²π²/(2L²)
    let g = Grid1D::new(-1.0, 3.0, 4000);
    let v = g.sample(|x| if (0.0..=2.0).contains(&x) { 0.0 } else { 1e6 });
    let b = EigenBasis::numeric(g, &v, 3);
    for (k, e) in b.energies.iter().enumerate() {
        let want = well_energy(k + 1, 2.0);
        assert!((e - want).abs() / want < 5e-3, "E_{} = {e} vs {want}", k + 1);
    }
}

#[test]
fn coherent_state_stays_gaussian_and_oscillates() {
    // PS10 40: ⟨x⟩(t) = √2 |α| cos(ωt − arg α), σ_x = 1/√2 for ω = 1
    let g = Grid1D::new(-15.0, 15.0, 1024);
    let basis = Arc::new(EigenBasis::harmonic(g, 60, 1.0, 0.0));
    let alpha = c64(2.0, 0.0);
    let mut e = EigenExpansion::coherent(basis.clone(), alpha);
    assert!(close(e.captured(), 1.0, 1e-10));
    for k in 0..=8 {
        let t = 0.4 * k as f64;
        e.set_time(t);
        let w = e.wave();
        let mo = w.moments();
        let x_cl = 2f64.sqrt() * 2.0 * t.cos();
        assert!(close(mo.x, x_cl, 1e-6), "⟨x⟩({t}) = {} vs {x_cl}", mo.x);
        assert!(close(mo.sigma_x, 0.5f64.sqrt(), 1e-6), "σ_x({t}) = {}", mo.sigma_x);
        assert!(close(mo.uncertainty_product(), 0.5, 1e-6));
        // still Gaussian: compare with the shifted ground-state Gaussian with momentum ⟨p⟩
        let gauss = Wave1D::gaussian(g, mo.x, 0.5f64.sqrt(), mo.p);
        let overlap = gauss.inner(&w).norm();
        assert!(close(overlap, 1.0, 1e-6), "overlap with a Gaussian {overlap}");
    }
    // the same with the split-step propagator in V = x²/2
    let v = g.sample(|x| 0.5 * x * x);
    let mut w = EigenExpansion::coherent(basis, alpha).wave();
    let mut prop = SplitStep::new(g, 0.005).with_potential(&v);
    prop.advance(&mut w, std::f64::consts::PI);
    assert!(close(w.mean_x(), -2.0 * 2f64.sqrt(), 1e-3), "⟨x⟩(π) = {}", w.mean_x());
}

#[test]
fn expansion_round_trip_and_energy_collapse() {
    let g = Grid1D::new(-12.0, 12.0, 512);
    let basis = Arc::new(EigenBasis::harmonic(g, 40, 1.0, 0.0));
    let w = Wave1D::gaussian(g, 1.0, 0.9, 0.5);
    let e = EigenExpansion::project(basis.clone(), &w);
    assert!(close(e.captured(), 1.0, 1e-8), "captured {}", e.captured());
    let back = e.wave();
    assert!(close(back.inner(&w).norm(), 1.0, 1e-8));
    let mut rng = QbRng::seed(7);
    let mut counts = vec![0usize; 40];
    for _ in 0..20000 {
        let mut c = e.clone();
        let n = measure_energy(&mut c, &mut rng).unwrap();
        counts[n] += 1;
        assert!(close(c.probabilities()[n], 1.0, 1e-12));
    }
    let p = e.probabilities();
    for n in 0..4 {
        assert!(
            close(counts[n] as f64 / 20000.0, p[n], 0.015),
            "P({n}): {} vs {}",
            counts[n] as f64 / 20000.0,
            p[n]
        );
    }
}

#[test]
fn rectangular_barrier_transmission() {
    // PS10 39: T = 1/(1 + V² sinh²(κa)/(4E(V−E)))
    let (v0, a) = (1.0, 1.0);
    let barrier = Piecewise::barrier(0.0, a, v0);
    for &e in &[0.1, 0.3, 0.5, 0.8, 0.99, 1.2, 2.0, 4.0] {
        let exact = rect_barrier_transmission(e, v0, a);
        let tm = barrier.scatter(e);
        assert!(
            close(tm.transmission, exact, 1e-10),
            "transfer matrix T({e}) = {} vs {exact}",
            tm.transmission
        );
        assert!(close(tm.transmission + tm.reflection, 1.0, 1e-10));
        let nu = numerov_scatter(-5.0, 5.0, 0.001, |x| barrier.value(x), e);
        assert!(
            close(nu.transmission, exact, 2e-3),
            "Numerov T({e}) = {} vs {exact}",
            nu.transmission
        );
        assert!(close(nu.transmission + nu.reflection, 1.0, 2e-3));
        // the two wavefunctions agree
        let x = 0.5;
        let i = ((x + 5.0) / 0.001f64).round() as usize;
        assert!((nu.psi[i] - tm.psi(x)).norm() < 5e-3, "ψ({x})");
    }
    // transmission resonance above the barrier: k a = π ⇒ T = 1
    let e_res = v0 + std::f64::consts::PI.powi(2) / (2.0 * a * a);
    assert!(close(barrier.scatter(e_res).transmission, 1.0, 1e-10));
}

#[test]
fn position_measurement_statistics_and_collapse() {
    let g = Grid1D::new(-20.0, 20.0, 1024);
    let w = Wave1D::gaussian(g, 2.0, 1.5, 0.0);
    let mut rng = QbRng::seed(1);
    let mut h = ShotHistogram::binned(-6.0, 10.0, 32);
    h.add_all(&born_samples_x(&w, 50_000, &mut rng));
    assert!(
        close(h.mean(), 2.0, 0.03) && close(h.std(), 1.5, 0.03),
        "mean {} std {}",
        h.mean(),
        h.std()
    );
    let pred = h.predict_from_density(&w.xs(), &w.density());
    let frac = h.fractions();
    for b in 0..h.len() {
        assert!(close(frac[b], pred[b], 0.006), "bin {b}: {} vs {}", frac[b], pred[b]);
    }
    // collapse: narrow detector ⇒ σ_x ≈ w, σ_p ≈ 1/(2w)
    let mut c = w.clone();
    let x0 = measure_position(&mut c, 0.2, &mut rng);
    let mo = c.moments();
    assert!(close(c.norm_sqr(), 1.0, 1e-12));
    assert!(
        close(mo.x, x0, 0.05) && mo.sigma_x < 0.21,
        "x0 {x0}, ⟨x⟩ {}, σ {}",
        mo.x,
        mo.sigma_x
    );
    assert!(mo.sigma_p > 2.0, "σ_p {}", mo.sigma_p);
    // momentum measurement
    let mut c = w.clone();
    let p0 = measure_momentum(&mut c, 0.2, &mut rng);
    assert!(close(c.mean_p(), p0, 0.02));
    assert!(c.sigma_x() > 2.5, "σ_x after a momentum measurement {}", c.sigma_x());
    // region measurement
    let mut yes = 0;
    for _ in 0..4000 {
        let mut c = w.clone();
        if measure_region(&mut c, 2.0, 30.0, &mut rng) {
            yes += 1;
            assert!(c.prob_between(2.0, 30.0) > 0.999);
        }
    }
    assert!(close(yes as f64 / 4000.0, 0.5, 0.03), "P(x > 2) ≈ {}", yes as f64 / 4000.0);
}

#[test]
fn free_spreading_prediction_from_moments() {
    // σ_x²(t + dt) = σ_x² + 2 cov dt + σ_p² dt² holds for any free state, e.g. after a collapse
    let g = Grid1D::new(-80.0, 80.0, 4096);
    let mut w = Wave1D::gaussian(g, 0.0, 1.0, 0.5);
    let mut prop = SplitStep::new(g, 0.02);
    assert!(close(w.moments().cov_xp, 0.0, 1e-9));
    prop.advance(&mut w, 3.0);
    let m3 = w.moments();
    assert!(m3.cov_xp > 0.0, "a spreading packet is chirped: cov {}", m3.cov_xp);
    assert!(close(m3.free_sigma_x(2.0), free_gaussian_sigma(1.0, 5.0), 1e-6));
    collapse_position(&mut w, 1.2, 0.6);
    let mc = w.moments();
    prop.advance(&mut w, 4.0);
    assert!(
        close(w.sigma_x(), mc.free_sigma_x(4.0), 1e-5),
        "{} vs {}",
        w.sigma_x(),
        mc.free_sigma_x(4.0)
    );
}
