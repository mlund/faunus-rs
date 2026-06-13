# Running faunus as a web (WASM) and/or native cross-platform app

Notes for an `eframe`/`egui` front-end that drives the faunus simulation core.
**On hold** — see the native egui app effort first.

## Native vs. web — pick by goal

`eframe` builds **both** from one egui codebase:

| | Native eframe (per-OS binary) | WASM/web (one artifact) |
|---|---|---|
| Win/Mac/Linux | yes — one binary each (CI matrix) | yes — one `.wasm`, any browser, **also covers mobile** |
| Install | per-OS download | none (open a URL) |
| Speed | **full native** | ~2–10× slower, single-threaded unless SharedArrayBuffer |
| Faunus features | everything (threads, fs, `cli`, umbrella/WL/Gibbs) | constrained (see blockers) |
| Effort | CI build + signing | the port below, then trivial to host |

**WASM is a browser target, not an OS target** — one `.wasm` already runs on
Windows/macOS/Linux/mobile with zero install. For a compute-heavy MC sim,
**native is the performance path**; web is the zero-install/demo/teaching path.
Recommendation: native-first (CI matrix), add web as a companion.

## Faunus is well-suited

- **Core MC loop is single-threaded** — no `rayon`. Parallelism is only
  `std::thread::scope` in optional features (`umbrella.rs`, `wang_landau.rs`,
  `cli.rs` replicas, `montecarlo/gibbs.rs`). A single run needs no
  SharedArrayBuffer / COOP-COEP.
- **`MarkovChain::run_n_steps(n)`** (`montecarlo/mod.rs:407`) and `.iter()` —
  drive the sim in **chunks from `eframe::App::update`** (e.g. `run_n_steps(2000)`
  per frame, repaint). Responsive UI + live plots, no Web Worker.
- **Input is trivial** — a dropped file is a string; faunus's Jinja render already
  works on a string (`auxiliary.rs` `render_str`). bytes → UTF-8 → render+parse.
- **Live results need no file IO** — analyses accumulate in `BlockAverage` and
  expose `to_yaml`/means; plot straight from the analysis objects each frame.

## The `cli` feature already carves out the native parts

`[features] default = ["cli"]`. It gates (via `#[cfg(feature = "cli")]`):
`cli.rs`, `umbrella.rs`, `wang_landau.rs`, `topology/io/frame_state`, plus
`clap`/`pretty_env_logger`/`indicatif`. **`cargo build -p faunus
--no-default-features` compiles clean.** So: **WASM = depend on faunus with
`default-features = false`.**

## Remaining wasm32-unknown-unknown items (NOT covered by `cli`)

| item | where | impact | fix |
|---|---|---|---|
| `getrandom 0.2` (via `rand`) | Cargo.lock | **won't compile** without `js` | target-gate `getrandom = { version = "0.2", features = ["js"] }` |
| `std::time::Instant` | `propagate/mod.rs:139` (run loop), `time.rs`, `hamiltonian.rs` | compiles, **panics at runtime** | `web-time` drop-in `Instant`; alias in `time.rs` |
| `gibbs.rs` `thread::scope` | `pub mod gibbs;` — **not** behind `cli` | panics only if a Gibbs ensemble runs | gate it (fold into `cli`/a `parallel` feature) or don't use Gibbs |
| fs output writers | `auxiliary.rs:115` `open_compressed`, `topology/io/xyz.rs` | `File::create` returns `Err` (no panic) | see Output |

Two hard blockers: **getrandom-js** (compile) and **web-time** (the `Instant`
panic). Plus gating `gibbs`. The `cli` feature removed the rest.

## Output

`open_compressed` already returns `Box<dyn Write + Send>` (`auxiliary.rs:115`), so
the streaming writers are trait-objects.
- **Simplest (web)**: run analyses with **no `file:`** — results live in
  `BlockAverage`, read via `to_yaml`/means for the UI. Nothing touches fs.
- **For downloads**: swap the `Box<dyn Write>` target from `File` to an in-memory
  `Vec<u8>`/`Cursor`, collect, hand to `rfd` (= browser download).

## File-IO crate picks (tuned to faunus)

- **Drag-in**: no `rfd` needed — egui surfaces dropped files via
  `ctx.input(|i| i.raw.dropped_files.clone())`; on web each `DroppedFile` carries
  `.bytes`. Read directly.
- **Out**: `rfd` (download). In-memory buffers, not a VFS.
- **`vfs` / OPFS**: skip — you feed strings in and collect bytes out; no need for a
  runtime FS or durable storage for a "load → run → export" app.
- **`rust-embed` / `include_dir`**: useful — embed example inputs (the
  `double_layer` one is a perfect demo) + forcefield includes; expose a "load
  example" dropdown. NB: faunus YAML `include:` must resolve from embedded bytes on
  web, not the FS.
- **eframe `persistence`** (localStorage): UI/parameter state only.

## Suggested shape

`faunus` stays the lib. Thin `faunus-egui` crate:
```toml
faunus = { path = "../faunus", default-features = false }
[target.'cfg(target_arch = "wasm32")'.dependencies]
getrandom = { version = "0.2", features = ["js"] }
web-time = "1"
```
Native release = CI matrix (`windows`/`macos`/`ubuntu`) or `cargo-dist`
(builds + installers; mac needs codesign/notarize, Linux best as AppImage).
Web = `trunk build`, host static. UI + faunus-driving logic identical across both.

Net new work for a basic web build: ~2 Cargo lines, `web-time` aliasing in
`time.rs`, one `cfg`/feature gate on `gibbs`, analyses configured without `file:`.
The MC core compiles and runs unchanged.
