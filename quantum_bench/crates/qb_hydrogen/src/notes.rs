//! Notes of the hydrogen presets (markup: "# " headings, "- " bullets).

pub const NUMBERS: &str = "\
# The exercise
A hydrogen atom in a state with principal quantum number n (the preset starts with n = 4). Atomic units: lengths in a₀, energies E_n = −1/(2n²) Hartree = −13.6 eV/n²; angular momenta in units of ħ.
- (a) For a given n: l = 0, 1, …, n − 1 and m = −l, …, +l, so n² states share the energy E_n. For n = 4: l = 0 (m = 0), l = 1 (m = 0, ±1), l = 2 (m = 0, ±1, ±2), l = 3 (m = 0, ±1, ±2, ±3). The eigenvalues of l² are l(l+1)ħ² = 0, 2ħ², 6ħ², 12ħ²; those of l_z are mħ.
- (b) l± = l_x ± i l_y changes m by ±1: l± ψ_nlm = ħ√(l(l+1) − m(m ± 1)) ψ_nl,m±1 for normalised Y_lm (Condon–Shortley phases, as in Eq. 44.4). At the ends the ladder stops: l₊ψ_nll = 0 and l₋ψ_nl,−l = 0.

# Try
- Choose n, l, m in the panel, or click a sublevel in the level diagram: all n² states of a shell sit at the same energy.
- Step m with l₊ and l₋: the factor ħ√(l(l+1) − m(m ± 1)) is shown; at m = ±l the result is 0 (the state is kept).
- 3D view: |ψ_nlm|² does not depend on φ. For m ≠ 0 the orbital is a ring around z whose phase (colour) winds m times around it; for m = 0, ψ is real (up to the global phase) and its angular nodes are cones around the z axis or the x–y plane.
- Turn on the cut plane: the n − l − 1 radial nodes appear as dark shells; compare with the radial panel.
- Measure l², l_z or E: for an eigenstate the outcome is certain and the state does not change. Measure the position (M), reset (R), measure again …: the dots build up |ψ|². Repeat N does it in one go.

# Behind the scenes
ψ_nlm = R_nl(r) Y_lm(θ, φ) with associated Laguerre and Legendre polynomials for all n ≤ 4, evaluated analytically on the GPU while ray-marching a sphere that scales with the orbital (≈ n²). Brightness and opacity are relative to a scale ρ_ref per state (between the maximum of |ψ|² and the density where most of the probability is), so that small and large orbitals and the outer shells of ns states are all visible; the densest parts saturate. For a stationary state the time evolution only turns the global phase e^(−iE_n t/ħ): the colours rotate, the density stays. Time in ħ/E_h = 24.2 as, slowed down.";

pub const RADIUS: &str = "\
# The exercise
- (a) E_n = −13.6 eV/n² depends only on n (degenerate in l and m, as long as spin, fine structure and external fields are ignored). n sets the shell (energy, size ∝ n²), l the orbital angular momentum (l² = l(l+1)ħ²), m its projection on the quantisation axis z (l_z = mħ).
- (b) With the classical expression, Ê = −½ e²/(4πε₀ r̂), in atomic units −½·(1/r): ⟨Ê⟩ = −½⟨1/r⟩ = −1/(2n²) Hartree = E_n for (1, 0, 0) and (2, 1, 0). (It works for every eigenstate because of the virial theorem, ⟨V⟩ = 2E_n.)
- Lyman α (1s ↔ 2p): ΔE = E₂ − E₁ = 10.2 eV, λ = hc/ΔE = 121.6 nm (vacuum ultraviolet; measured 121.57 nm).
- (c) Ground state: ⟨r⟩ = 3a₀/2 and ⟨r²⟩ = 3a₀². In general ⟨r⟩ = (3n² − l(l+1))a₀/2, ⟨r²⟩ = n²(5n² + 1 − 3l(l+1))a₀²/2, ⟨1/r⟩ = 1/(n²a₀), ⟨1/r²⟩ = 1/((l + ½)n³a₀²) (needed in PS12 48 c).
- (d) The 1s density is spherically symmetric: ⟨x⟩ = 0 and ⟨x²⟩ = ⟨r²⟩/3 = a₀².

# Try
- Switch between ψ₁₀₀ and ψ₂₁₀: the table compares the numerical integrals with the closed forms.
- Radial panel: r²|R|² (filled) is the probability density in r. Its maximum, the most probable r (a₀ for 1s), is not the mean ⟨r⟩ = 1.5 a₀: the long tail pulls the mean outwards.
- Repeat N position measurements: the histogram of |r| follows r²|R|², and the mean of the outcomes approaches ⟨r⟩.
- Click other levels: ⟨1/r⟩ = 1/n² for every l, while ⟨r⟩ shrinks with growing l at fixed n.

# Behind the scenes
Expectation values come from radial integrals ∫R_nl R_n'l' r^(2+p) dr (Gauss–Legendre quadrature) and angular integrals of the Y_lm; the closed forms are shown next to them.";

pub const LY: &str = "\
# The exercise
Ψ(r, 0) = αψ₂₁₁ + βψ₂₁₋₁ with |α|² + |β|² = 1 and Y₁,±₁ = ∓√(3/8π) sin θ e^(±iφ) (Eq. 44.4).
- (a) Eigenfunction of l_y = −iħ(cos φ ∂_θ − cot θ sin φ ∂_φ): this needs α = β, so Ψ = (ψ₂₁₁ + ψ₂₁₋₁)/√2 = −i √(3/4π) R₂₁(r) sin θ sin φ, with the eigenvalue 0 (m_y = 0). It is the p_y orbital: two lobes along the y axis, of opposite sign.
- (b) l_z Ψ = −iħ ∂_φ Ψ ∝ sin θ cos φ is not proportional to Ψ: no l_z eigenstate. A measurement of l_z gives +ħ or −ħ, each with probability ½ (bars |c_m|²).
- (c) Both components have the energy E₂, so Ψ(r, t) = Ψ(r, 0) e^(−iE₂t/ħ): only a global phase turns and |Ψ|² is stationary.

# Try
- Play: all colours (the phase) rotate together, the shape stays (part c).
- Measure l_y: m_y = 0 with certainty, the state is unchanged. Measure l_z (M): ±ħ at random, and the cloud becomes the torus ψ₂₁₊₁ or ψ₂₁₋₁ (phase winding once around z). Measuring l_y on that torus gives −ħ, 0, +ħ with probabilities ¼, ½, ¼.
- Relative phase δ (β = |β| e^(iδ)): the lobes turn about z by δ/2. δ = 0 is p_y, δ = 180° is p_x ∝ ψ₂₁₋₁ − ψ₂₁₁ (an l_x eigenstate, no longer an l_y one).
- |α|² = 1: the single ψ₂₁₁, a torus around z with phase e^(iφ); in between, an elongated ring.

# Behind the scenes
Measurements are projections in the basis of the 30 states with n ≤ 4; l_y is diagonalised in each l block (eigenvectors in the Condon–Shortley convention). Time in ħ/E_h, slowed down.";

pub const DIPOLE: &str = "\
# Superposition of 1s and 2p
Ψ = c₁ψ₁₀₀ + c₂ψ₂₁ₘ mixes two energies, E₁ = −13.6 eV and E₂ = −3.4 eV. Now the relative phase turns, e^(−i(E₂ − E₁)t/ħ), and the density |Ψ|² = |c₁ψ₁₀₀|² + |c₂ψ₂₁ₘ|² + 2 Re(c₁* c₂ ψ₁₀₀ ψ₂₁ₘ e^(−iωt)) oscillates at the Bohr frequency ω = (E₂ − E₁)/ħ (10.2 eV/ħ, period 0.41 fs; slowed down here).
- m = 0: the charge sloshes along z. ⟨z⟩(t) = 2|c₁c₂| ⟨1s|z|2p₀⟩ cos(ωt − δ) with ⟨1s|z|2p₀⟩ = 128√2/243 a₀ = 0.745 a₀; ⟨x⟩ = ⟨y⟩ = 0.
- m = ±1: the charge rotates about z (counter-clockwise seen from +z for m = +1, clockwise for m = −1); ⟨x⟩ and ⟨y⟩ oscillate 90° out of phase on a circle of radius 2|c₁c₂|·0.745/√2 a₀, ⟨z⟩ = 0.

# Radiation (semiclassical picture)
An oscillating dipole d = −e⟨r⟩(t) radiates at ω, i.e. at the Lyman-α wavelength 121.6 nm. A dipole oscillating along z (Δm = 0) emits π light: linearly polarised along z, nothing along z. A dipole rotating about z (Δm = ±1) emits σ± light: circularly polarised seen along z, linearly polarised seen from the side (compare PS12 47). This is the semiclassical picture, in which the expectation value radiates like an antenna: it gives the frequencies, polarisations and selection rules. In it, a stationary state does not radiate; the spontaneous decay of 2p needs the quantised field.

# Try
- Play and watch the cloud and ⟨z⟩(t); switch the 2p sublevel to m = +1 or −1: rotation instead of sloshing.
- Mixing |c₂|²: the amplitude is ∝ |c₁c₂|, largest at ½. The phase δ shifts the oscillation in time.
- Measure E (M): the state collapses onto ψ₁₀₀ or ψ₂₁ₘ, the oscillation stops.";

pub const ZEEMAN: &str = "\
# The exercise
Levels with n₁, l = 1 (lower) and n₂ > n₁, l = 2 (upper) in a magnetic field B along z. Normal Zeeman effect: orbital magnetic moment only, no spin (as in the exercise; real hydrogen has spin and fine structure, which give the anomalous Zeeman effect). Each level splits into 2l + 1 equidistant sublevels, E = E_n + μ_B B m with μ_B = eħ/2m_e = 57.9 µeV/T.
- Electric-dipole selection rules: Δl = ±1 and Δm = 0, ±1, with Δm = m(upper) − m(lower). Here: 3 + 3 + 3 = 9 transitions.
- All transitions with the same Δm have the same photon energy hν₀ + Δm μ_B B: three lines.
- Polarisation: π (Δm = 0) linearly polarised along B; σ± (Δm = ±1) circularly polarised seen along B (photon angular momentum ±ħ along B), linearly polarised ⊥ B seen from the side. Seen along B the π line is missing.

# Try
- B slider: the sublevels fan out linearly. The diagram exaggerates the splitting; the spectrum is to scale (Δν = μ_B B/h = 14.0 GHz per tesla).
- Click a transition (arrow or list): the 3D view shows (ψ₂₁ₘ₁ + ψ₃₂ₘ₂)/√2, whose charge oscillates at the transition frequency, along z for π and rotating about z for σ± (dipole arrow and ⟨x⟩, ⟨y⟩, ⟨z⟩ plot).
- Seen ⊥ B / ∥ B: intensities σ⁻ : π : σ⁺ = 1 : 2 : 1 from the side, π absent along B (classical dipole radiation, equally populated upper sublevels).

# Behind the scenes
Hydrogen with n₁ = 2 (2p) and n₂ = 3 (3d): the Hα line, λ₀ = 656 nm with E_n = −13.6 eV/n². The 3D oscillation is slowed down; the Zeeman shift (3·10⁻⁵ of the photon energy at 1 T) is included in the phases but not visible there.";
