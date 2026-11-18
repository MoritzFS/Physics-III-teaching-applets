//! Shared widgets. All of them honour presentation mode and light/dark
//! through [`crate::Look`].
//!
//! | widget | for |
//! |---|---|
//! | [`PlotFrame`], [`PlotTransform`], [`PlotResponse`] | axes, pan/zoom, data ↔ screen (base of all plots) |
//! | [`WavePlot`], [`WaveCurve`], [`Marker`], [`WaveStyle`], [`Domain`] | ψ(x) or Φ(p) with V(x), phase colouring, markers |
//! | [`ReprView`], [`view_bar`] | the x / p / both switch, display modes, layout |
//! | [`LevelLadder`] | clickable levels with |c_n|² and phasors |
//! | [`HistogramPlot`] | measured counts vs Born rule |
//! | [`TimeSeries`], [`TimeSeriesPlot`] | σ_x(t), P(t), … |
//! | [`Heatmap2D`], [`field_image`] | complex 2D fields |
//! | [`MeasureFlash`] | the collapse effect |
//! | [`phase_color`], [`PHASE_WGSL`], [`phase_wheel`] | the phase colour (CPU + GPU) |
//! | [`readout`], [`readout_row`], [`fmt_num`], [`repeat_button`] | live values, "repeat N" |

pub mod color;
pub mod flash;
pub mod heatmap;
pub mod histogram;
pub mod ladder;
pub mod plot;
pub mod readout;
pub mod repr;
pub mod timeseries;
pub mod wave_plot;

pub use color::{PHASE_WGSL, complex_color, density_color, density_rgb, phase_color, phase_color_alpha, phase_rgb, phase_wheel};
pub use flash::{FlashKind, MeasureFlash};
pub use heatmap::{Heatmap2D, HeatmapMode, field_image};
pub use histogram::{HistScale, HistogramPlot};
pub use ladder::{LadderResponse, LevelLadder};
pub use plot::{PanZoom, PlotFrame, PlotResponse, PlotTransform, label_box, nice_ticks, plot_dashed, plot_line, tick_label};
pub use readout::{fmt_num, readout, readout_row, repeat_button};
pub use repr::ReprView;
pub use timeseries::{TimeSeries, TimeSeriesPlot, TimeWindow};
pub use wave_plot::{Domain, Marker, WaveCurve, WavePlot, WaveStyle, draw_curve, segmented, view_bar};
