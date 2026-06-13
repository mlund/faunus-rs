//! Native egui front-end for the Guldbrand electric double-layer simulation.
//!
//! Set the physical inputs (surface charge density, box dimensions, the 2 Na⁺ ⇌ Ca²⁺
//! swap free energy `dG`, and the number of MC sweeps), press Start, and watch the
//! osmotic pressure, ion molarities, and density profiles converge live. The simulation
//! engine is the `faunus` library, driven on the main thread in chunks so the UI stays
//! responsive.

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use faunus::analysis::{Analyze, DoubleLayerPressure, DoubleLayerPressureBuilder};
use faunus::backend::{get_medium, Backend};
use faunus::histogram::Histogram;
use faunus::montecarlo::MarkovChain;
use faunus::propagate::Propagate;
use faunus::cell::Shape;
use faunus::group::GroupCollection;
use faunus::selection::Selection;
use faunus::{ParticleSystem, WithCell, MOLAR_TO_INV_ANGSTROM3, R_IN_KJ_PER_MOL};

const EPS_R: f64 = 78.05; // ε_r·T = 2.327e4 K at 298.15 K (Guldbrand water)
const TEMPERATURE: f64 = 298.15;
const COULOMB_PREFACTOR: f64 = 1389.3545764438197; // TO_CHEMISTRY_UNIT, kJ·Å/mol
const N_BINS: usize = 100;
const SWEEPS_PER_FRAME: usize = 80; // chunk size; tune for responsiveness
const EQUIL_FRACTION: f64 = 0.2; // discard this fraction of the run before sampling

/// Physical inputs the user controls.
#[derive(Clone, PartialEq)]
struct Inputs {
    sigma: f64,   // surface charge density |σ|, e/Å²
    lxy: f64,     // lateral box length, Å
    lz: f64,      // plate separation 2a, Å
    dg: f64,      // 2 Na⁺ ⇌ Ca²⁺ free energy, kJ/mol
    n_sweeps: usize,
}

impl Default for Inputs {
    fn default() -> Self {
        // Guldbrand monovalent reference: σ₀=0.2244 C/m², 80 ions, 2a=21 Å, high dG ⇒ pure Na.
        Self { sigma: 0.014006, lxy: 53.44, lz: 21.0, dg: 20.0, n_sweeps: 20000 }
    }
}

/// Parameters derived from the inputs via electroneutrality (2·σ·area = total charge).
struct Derived {
    total_charge: usize, // = number of monovalent ions; multiple of 4
    sigma_eff: f64,      // n/(2·Lxy²), so the walls exactly neutralize the integer ions
    k: usize,            // counterions per half
    ca_slots: usize,     // max Ca²⁺ per half
}

impl Derived {
    fn from(inp: &Inputs) -> Self {
        let raw = 2.0 * inp.sigma * inp.lxy * inp.lxy;
        let total_charge = (((raw / 4.0).round() as usize) * 4).max(4);
        let sigma_eff = total_charge as f64 / (2.0 * inp.lxy * inp.lxy);
        Self { total_charge, sigma_eff, k: total_charge / 2, ca_slots: total_charge / 4 }
    }
}

/// A concrete (no-Jinja) double-layer input YAML built from the inputs.
fn make_yaml(inp: &Inputs, d: &Derived) -> String {
    let c_wall = -4.0 * COULOMB_PREFACTOR / EPS_R * d.sigma_eff;
    let cutoff = (inp.lxy * 10.0 * 1000.0).round() / 1000.0;
    let half_box = inp.lxy / 2.0;
    let half_lz = inp.lz / 2.0;
    let dp = ((inp.lz / 4.0) * 100.0).round() / 100.0;
    let spec_weight = (d.total_charge / 10).max(1);
    format!(
        r#"atoms:
  - {{ name: Na, charge: 1.0, sigma: 0.0, mass: 23.0 }}
  - {{ name: Ca, charge: 2.0, sigma: 0.0, mass: 40.0 }}
molecules:
  - {{ name: na_lo, atoms: [Na] }}
  - {{ name: na_up, atoms: [Na] }}
  - {{ name: ca_lo, atoms: [Ca] }}
  - {{ name: ca_up, atoms: [Ca] }}
system:
  cell: !Slit [{lxy}, {lxy}, {lz}]
  medium: {{ permittivity: !Fixed {eps_r}, temperature: {temp} }}
  energy:
    nonbonded:
      default:
        - !Coulomb {{ cutoff: {cutoff} }}
    customexternal:
      - selection: "all"
        constants: {{ c: {c_wall}, b: {half_box}, hg: {half_lz} }}
        function: >-
          c * q * (
          2*b*asinh(b/sqrt(b*b+(z+hg)*(z+hg)))
          - (z+hg)*atan(b*b/((z+hg)*sqrt(2*b*b+(z+hg)*(z+hg))))
          + 2*b*asinh(b/sqrt(b*b+(hg-z)*(hg-z)))
          - (hg-z)*atan(b*b/((hg-z)*sqrt(2*b*b+(hg-z)*(hg-z))))
          )
      - {{ selection: "molecule na_lo ca_lo", function: "1000000.0 if z > 0 else 0.0" }}
      - {{ selection: "molecule na_up ca_up", function: "1000000.0 if z < 0 else 0.0" }}
  blocks:
    - {{ molecule: na_lo, N: {k}, active: {k}, insert: !RandomAtomPos {{ directions: xy }} }}
    - {{ molecule: na_up, N: {k}, active: {k}, insert: !RandomAtomPos {{ directions: xy }} }}
    - {{ molecule: ca_lo, N: {ca_slots}, active: 0, insert: !RandomAtomPos {{ directions: xy }} }}
    - {{ molecule: ca_up, N: {ca_slots}, active: 0, insert: !RandomAtomPos {{ directions: xy }} }}
propagate:
  seed: !Fixed 42
  criterion: Metropolis
  repeat: {n_sweeps}
  collections:
    - !Stochastic
      repeat: {total_charge}
      moves:
        - !TranslateMolecule {{ molecule: na_lo, dp: {dp}, weight: {k} }}
        - !TranslateMolecule {{ molecule: na_up, dp: {dp}, weight: {k} }}
        - !TranslateMolecule {{ molecule: ca_lo, dp: {dp}, weight: {ca_slots} }}
        - !TranslateMolecule {{ molecule: ca_up, dp: {dp}, weight: {ca_slots} }}
        - !SpeciationMove
          temperature: {temp}
          weight: {spec_weight}
          reactions:
            - ["na_lo + na_lo = ca_lo", !dG {dg}]
            - ["na_up + na_up = ca_up", !dG {dg}]
"#,
        lxy = inp.lxy,
        lz = inp.lz,
        eps_r = EPS_R,
        temp = TEMPERATURE,
        cutoff = cutoff,
        c_wall = c_wall,
        half_box = half_box,
        half_lz = half_lz,
        k = d.k,
        ca_slots = d.ca_slots,
        n_sweeps = inp.n_sweeps,
        total_charge = d.total_charge,
        dp = dp,
        spec_weight = spec_weight,
        dg = inp.dg,
    )
}

/// A live simulation: the Markov chain plus the analysis state the GUI accumulates.
struct Sim {
    mc: MarkovChain<Backend>,
    dlp: DoubleLayerPressure,
    sel_na: Selection,
    sel_ca: Selection,
    rho_na: Histogram, // counts per z-bin, summed over samples
    rho_ca: Histogram,
    count_na_sum: f64,
    count_ca_sum: f64,
    samples: usize,
    bin_width: f64,
    area: f64,
    volume: f64,
    steps_done: usize,
    n_sweeps: usize,
    equil: usize,
}

impl Sim {
    fn build(inp: &Inputs, d: &Derived) -> anyhow::Result<Self> {
        let yaml = make_yaml(inp, d);
        let tmp = tempfile::NamedTempFile::new()?;
        std::fs::write(tmp.path(), yaml.as_bytes())?;
        let path = tmp.path();
        let mut rng = rand::thread_rng();
        let context = Backend::new(path, None, &mut rng)?;
        let medium = get_medium(path)?;
        let rt = R_IN_KJ_PER_MOL * TEMPERATURE;
        let propagate = Propagate::from_file(path, &context)?;
        let mc = MarkovChain::new(context, propagate, rt, vec![])?;

        let builder: DoubleLayerPressureBuilder =
            serde_yml::from_str("selection: \"atomtype Na Ca\"\ndensity_bins: 50\nfrequency: !Every 1\n")?;
        let dlp = builder.build(mc.context(), Some(&medium))?;

        let bbox = mc
            .context()
            .cell()
            .bounding_box()
            .ok_or_else(|| anyhow::anyhow!("cell has no bounding box"))?;
        let area = bbox.x * bbox.y;
        let half_gap = bbox.z / 2.0;
        let bin_width = bbox.z / N_BINS as f64;
        let volume = mc.context().cell().volume().unwrap_or(f64::INFINITY);

        Ok(Self {
            sel_na: Selection::parse("atomtype Na")?,
            sel_ca: Selection::parse("atomtype Ca")?,
            rho_na: Histogram::new(-half_gap, half_gap, bin_width)?,
            rho_ca: Histogram::new(-half_gap, half_gap, bin_width)?,
            count_na_sum: 0.0,
            count_ca_sum: 0.0,
            samples: 0,
            bin_width,
            area,
            volume,
            steps_done: 0,
            n_sweeps: inp.n_sweeps,
            equil: (inp.n_sweeps as f64 * EQUIL_FRACTION) as usize,
            mc,
            dlp,
        })
    }

    /// Sample the pressure analysis and accumulate per-species density + molarity.
    fn sample(&mut self) -> anyhow::Result<()> {
        self.dlp.perform_sample(self.mc.context(), self.steps_done, 1.0)?;
        let ctx = self.mc.context();
        for group in ctx.groups() {
            for i in group.iter_active() {
                let z = ctx.position(i).z;
                if ctx.atom_charge(i) < 1.5 {
                    self.rho_na.add(z);
                } else {
                    self.rho_ca.add(z);
                }
            }
        }
        self.count_na_sum += ctx.resolve_atoms_live(&self.sel_na).len() as f64;
        self.count_ca_sum += ctx.resolve_atoms_live(&self.sel_ca).len() as f64;
        self.samples += 1;
        Ok(())
    }

    /// Density profile of a species (bin-center z, density in mM) averaged over samples.
    fn profile(&self, hist: &Histogram) -> Vec<[f64; 2]> {
        if self.samples == 0 {
            return Vec::new();
        }
        let scale = 1e3 / (self.samples as f64 * self.bin_width * self.area * MOLAR_TO_INV_ANGSTROM3);
        hist.iter().map(|(z, count)| [z, count * scale]).collect()
    }

    fn molarity_mm(&self, count_sum: f64) -> f64 {
        if self.samples == 0 {
            return 0.0;
        }
        let mean = count_sum / self.samples as f64;
        1e3 * mean / (self.volume * MOLAR_TO_INV_ANGSTROM3)
    }
}

/// Read `{mean, error}` (mM) for a key from an already-built analysis report.
fn stat(report: &serde_yml::Value, key: &str) -> Option<(f64, f64)> {
    let m = report.get(key)?;
    Some((m.get("mean")?.as_f64()?, m.get("error")?.as_f64()?))
}

/// One labelled, tooltipped numeric input row in a 2-column grid; returns whether it changed.
fn input_field<N: egui::emath::Numeric>(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut N,
    speed: f64,
    range: std::ops::RangeInclusive<N>,
    tip: &str,
) -> bool {
    ui.label(label);
    let changed = ui
        .add(egui::DragValue::new(value).speed(speed).range(range))
        .on_hover_text(tip)
        .changed();
    ui.end_row();
    changed
}

#[derive(Default)]
struct App {
    inputs: Inputs,
    sim: Option<Sim>,
    running: bool,
    dirty: bool, // inputs changed since the current sim was built
    error: Option<String>,
}

impl App {
    fn start(&mut self) {
        if self.sim.is_none() || self.dirty {
            let d = Derived::from(&self.inputs);
            match Sim::build(&self.inputs, &d) {
                Ok(sim) => {
                    self.sim = Some(sim);
                    self.dirty = false;
                    self.error = None;
                }
                Err(e) => {
                    self.error = Some(e.to_string());
                    return;
                }
            }
        }
        self.running = true;
    }

    /// Advance the simulation by one frame's worth of sweeps.
    fn step(&mut self, ctx: &egui::Context) {
        let mut finished = false;
        let mut err = None;
        if let Some(sim) = self.sim.as_mut() {
            for _ in 0..SWEEPS_PER_FRAME {
                if sim.steps_done >= sim.n_sweeps {
                    finished = true;
                    break;
                }
                if let Err(e) = sim.mc.run_n_steps(1) {
                    err = Some(e.to_string());
                    break;
                }
                sim.steps_done += 1;
                if sim.steps_done > sim.equil {
                    if let Err(e) = sim.sample() {
                        err = Some(e.to_string());
                        break;
                    }
                }
            }
        }
        if finished || err.is_some() {
            self.running = false;
        }
        if let Some(e) = err {
            self.error = Some(e);
        }
        ctx.request_repaint();
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let d = Derived::from(&self.inputs);

        egui::TopBottomPanel::top("inputs").show(ctx, |ui| {
            ui.heading("Double Layer");
            ui.label("Osmotic pressure between two charged planes (Guldbrand et al. 1984).")
                .on_hover_text("Two uniformly charged walls with explicit point-charge counterions; the mono/divalent mix is set by the swap free energy dG.");
            egui::Grid::new("input_grid").num_columns(2).show(ui, |ui| {
                let mut changed = false;
                changed |= input_field(ui, "σ surface charge [e/Å²]", &mut self.inputs.sigma, 0.0001, 0.0001..=0.1,
                    "Surface charge density of each wall (0.014006 e/Å² = Guldbrand's 0.2244 C/m²).");
                changed |= input_field(ui, "x–y length [Å]", &mut self.inputs.lxy, 0.5, 10.0..=300.0,
                    "Lateral box side (periodic). With σ it fixes the ion count via electroneutrality.");
                changed |= input_field(ui, "z length (2a) [Å]", &mut self.inputs.lz, 0.5, 4.0..=60.0,
                    "Plate separation. Guldbrand: 21 Å (monovalent), 12 Å (divalent attraction).");
                changed |= input_field(ui, "dG (2Na⁺⇌Ca²⁺) [kJ/mol]", &mut self.inputs.dg, 0.5, -60.0..=60.0,
                    "Free energy of the charge-conserving swap. High → Na⁺; low → Ca²⁺; between → mixtures.");
                changed |= input_field(ui, "number of sweeps", &mut self.inputs.n_sweeps, 500.0, 1000..=2_000_000,
                    "MC sweeps (1 sweep = one move per ion). The first 20% are equilibration, not sampled.");
                if changed {
                    self.dirty = true;
                }
            });

            ui.separator();
            ui.horizontal(|ui| {
                ui.label(format!("total charge = {}", d.total_charge))
                    .on_hover_text("Total counterion charge = number of Na⁺ (or twice the number of Ca²⁺). 80 for the Guldbrand default.");
                ui.separator();
                ui.label(format!("σ_eff = {:.5} e/Å²", d.sigma_eff))
                    .on_hover_text("Effective surface charge after rounding the ion count to a multiple of 4 (keeps the cell electroneutral).");
            });

            ui.horizontal(|ui| {
                let label = if self.running { "⏸ Stop" } else { "▶ Start" };
                if ui
                    .button(label)
                    .on_hover_text("Start/resume or pause the simulation. Editing an input and pressing Start rebuilds.")
                    .clicked()
                {
                    if self.running {
                        self.running = false;
                    } else {
                        self.start();
                    }
                }
                let frac = self
                    .sim
                    .as_ref()
                    .map(|s| s.steps_done as f64 / s.n_sweeps.max(1) as f64)
                    .unwrap_or(0.0);
                ui.add(egui::ProgressBar::new(frac as f32).show_percentage())
                    .on_hover_text("Fraction of the requested sweeps completed.");
            });
            if let Some(err) = &self.error {
                ui.colored_label(egui::Color32::RED, err);
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(sim) = self.sim.as_ref() {
                ui.label(format!("samples: {}", sim.samples));
                let report = Analyze::<Backend>::to_yaml(&sim.dlp); // built once per frame
                egui::Grid::new("pressure_table").num_columns(2).striped(true).show(ui, |ui| {
                    let prow = |ui: &mut egui::Ui, name: &str, key: &str, hover: &str| {
                        ui.label(name).on_hover_text(hover);
                        let txt = report
                            .as_ref()
                            .and_then(|r| stat(r, key))
                            .map_or_else(|| "—".to_owned(), |(m, e)| format!("{m:.1} ± {e:.1} mM"));
                        ui.label(txt);
                        ui.end_row();
                    };
                    prow(ui, "p_ideal", "p_ideal/mM", "Entropic term kT·Σ C_i(0) — the midplane ion concentration.");
                    prow(ui, "p_corr", "p_corr/mM", "Configurational (electrostatic) term; its correlation part is the attraction.");
                    prow(ui, "p_osm (total)", "p_osm/mM", "Total osmotic/disjoining pressure. Positive = repulsive, negative = attractive.");
                    ui.end_row();
                    let mrow = |ui: &mut egui::Ui, name: &str, count_sum: f64, hover: &str| {
                        ui.label(name).on_hover_text(hover);
                        ui.label(format!("{:.1} mM", sim.molarity_mm(count_sum)));
                        ui.end_row();
                    };
                    mrow(ui, "⟨molarity⟩ Na⁺ (+1)", sim.count_na_sum, "Mean concentration of monovalent ions.");
                    mrow(ui, "⟨molarity⟩ Ca²⁺ (+2)", sim.count_ca_sum, "Mean concentration of divalent ions.");
                });

                ui.separator();
                ui.label("Density profiles ρ(z) [mM]")
                    .on_hover_text("Time-averaged ion number density across the gap; midplane at z=0, walls at z=±L_z/2.");
                Plot::new("density")
                    .legend(Legend::default())
                    .x_axis_label("z [Å]")
                    .y_axis_label("ρ [mM]")
                    .height(320.0)
                    .show(ui, |plot| {
                        plot.line(Line::new(PlotPoints::from(sim.profile(&sim.rho_na))).name("Na⁺ (+1)"));
                        plot.line(Line::new(PlotPoints::from(sim.profile(&sim.rho_ca))).name("Ca²⁺ (+2)"));
                    });
            } else {
                ui.label("Set the inputs above and press Start.");
            }
        });

        if self.running {
            self.step(ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_guldbrand_monovalent() {
        let d = Derived::from(&Inputs::default());
        assert_eq!(d.total_charge, 80, "Guldbrand default should be 80 charges");
        assert_eq!(d.k, 40);
        assert_eq!(d.ca_slots, 20);
    }

    #[test]
    fn build_run_and_report() {
        // Short run that still exercises build → step → sample → results extraction.
        let inp = Inputs { n_sweeps: 3000, ..Inputs::default() };
        let d = Derived::from(&inp);
        let mut sim = Sim::build(&inp, &d).expect("build");
        while sim.steps_done < sim.n_sweeps {
            sim.mc.run_n_steps(1).expect("step");
            sim.steps_done += 1;
            if sim.steps_done > sim.equil {
                sim.sample().expect("sample");
            }
        }
        assert!(sim.samples > 0);
        let report = Analyze::<Backend>::to_yaml(&sim.dlp).expect("report");
        let (p, e) = stat(&report, "p_osm/mM").expect("p_osm");
        assert!(p.is_finite() && e.is_finite(), "p_osm = {p} ± {e}");
        let na = sim.molarity_mm(sim.count_na_sum);
        let ca = sim.molarity_mm(sim.count_ca_sum);
        assert!(na > ca, "dG=20 should be Na-dominated: Na={na:.1} Ca={ca:.1} mM");
        let prof = sim.profile(&sim.rho_na);
        assert!(!prof.is_empty() && prof.iter().all(|p| p[1].is_finite()));
        eprintln!("p_osm = {p:.1} ± {e:.1} mM | Na = {na:.1} mM, Ca = {ca:.1} mM | samples = {}", sim.samples);
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Double Layer",
        options,
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
}
