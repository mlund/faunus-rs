# Umbrella Sampling

Umbrella sampling maps the potential of mean force (PMF) along a collective variable that the
system would otherwise cross too rarely to sample. Faunus divides the coordinate into overlapping
hard-wall windows and runs one walker per window in parallel. Each walker builds a density
histogram within its window, and adjacent windows are stitched into a continuous PMF from the
free-energy difference

$$\Delta F_{i \to i+1} = -RT \ln(f_i / f_{i+1}),$$

where $f_i$ is the fraction of window $i$'s samples that fall in the range the two windows share.

## Command line

```
faunus umbrella -i input.yaml [-s umbrella_states/] [-o pmf.csv] [-j 0]
```

| Flag   | Default            | Description                                    |
|--------|--------------------|------------------------------------------------|
| `-i`   | (required)         | YAML input file                                |
| `-s`   | `umbrella_states/` | Directory for per-window state and output files |
| `-o`   | `pmf.csv`          | PMF output file (CSV)                          |
| `-j`   | `0`                | Max parallel threads (`0` = all cores)         |

## YAML configuration

Add an `umbrella:` section alongside the usual `system:`, `propagate:`, and `analysis:` sections.
The `propagate:` section defines the Monte Carlo moves used in both the drive and the production
phase; the `analysis:` section runs once per window during production.

```yaml
umbrella:
  coordinate:
    property: mass_center_separation
    selections: ["molecule protein0", "molecule protein1"]
  windows:
    range: [25.0, 100.0]
    width: 10.0
    spacing: 6.0
    resolution: 1.0
  drive:
    force_constant: 5.0
```

### `coordinate` — collective variable

Any [collective variable](analysis.md#supported-properties) works; selections resolve separately
for each window.

### `windows` — window layout

| Key          | Required | Default | Description                              |
|--------------|----------|---------|------------------------------------------|
| `range`      | yes      |         | `[min, max]` of the CV range to cover    |
| `width`      | yes      |         | Full width of each window                |
| `spacing`    | yes      |         | Distance between adjacent window centers |
| `resolution` | no       | `1.0`   | Histogram bin width for the PMF          |

Window centers are placed at $c_i = \text{min} + \text{width}/2 + i \cdot \text{spacing}$.
Each window spans $[c_i - \text{width}/2,\; c_i + \text{width}/2]$.
Adjacent windows must overlap (`spacing < width`) for stitching.

### `drive` — drive phase

| Key              | Required | Description                                  |
|------------------|----------|----------------------------------------------|
| `force_constant` | yes      | Harmonic spring constant $k$ (kJ/mol/unit$^2$) |

During the drive phase a harmonic bias $\frac{1}{2}k(x - c_i)^2$ steers each walker toward the
center of its window. The drive ends as soon as the collective variable enters the window. If
`propagate.repeat` sweeps pass without the walker arriving, the run stops with an error;
strengthen `force_constant` or allow more sweeps.

## Per-window lifecycle

1. Drive — the harmonic bias pulls the walker into its target window. Skipped on restart if a
   state file already exists.
2. Production — the harmonic bias is switched off and a hard wall confines the walker to its
   window. The wall contributes zero energy inside the window and rejects any move that leaves
   it, so the sampled density is the unbiased Boltzmann density restricted to the window and needs
   no reweighting. The `propagate:` and `analysis:` sections run normally.

## Restart

Rerunning the same input loads any state files found in the `-s` directory, skips the drive phase,
and extends the existing histograms with further sampling. To drive a window afresh, delete its
state file.

## Output

### `pmf.csv`

One row per histogram bin:

| Column      | Description                                                       |
|-------------|-------------------------------------------------------------------|
| `cv`        | Bin center of the collective variable                             |
| `pmf_kT`    | Free energy in units of $RT$, shifted so that the minimum is zero |
| `stderr_kT` | Statistical uncertainty of `pmf_kT`                               |
| `spread_kT` | Disagreement between the windows covering the bin                 |

Where several windows cover a bin, their free energies are combined by inverse-variance weighting.
A window holding $N_j$ samples in the bin carries the counting uncertainty $RT/\sqrt{N_j}$, so the
windows combine to
$\sigma = RT / \sqrt{\sum_j N_j}$,
which shrinks as sampling accumulates.

Two effects make `stderr_kT` a lower bound on the true uncertainty. It assumes statistically
independent samples, whereas successive Monte Carlo sweeps are correlated; scale it by $\sqrt{g}$,
the statistical inefficiency of the collective variable, for a rigorous bound. It also measures a
single bin in isolation. The free-energy offset between adjacent windows is itself estimated from
the overlap fractions, and these offsets accumulate along the reaction coordinate. That
accumulated uncertainty is not propagated, so `stderr_kT` understates the error on free-energy
differences between distant bins.

`spread_kT` measures how far the individual windows covering a bin disagree about its free energy,
weighted by their counts. Read it as a stitching diagnostic rather than an error bar: it vanishes
wherever a single window covers the bin, and it is small wherever overlapping windows agree. Large
values flag poor overlap or unconverged windows. Warning: where `spread_kT` approaches $RT$, the
stitched PMF is unreliable — widen the windows or sample longer.

### `umbrella_states/window{i}/`

One self-contained directory per window, holding `state.yaml` (simulation state), `output.yaml`
(energy summary and move statistics), `histogram.yaml` (sampled CV counts), and any analysis files
produced by the input.
