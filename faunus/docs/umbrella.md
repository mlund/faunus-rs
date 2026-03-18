# Umbrella Sampling

Multi-walker umbrella sampling with free-energy stitching from overlap fractions.
The reaction coordinate is divided into overlapping hard-wall windows that are
run in parallel. A PMF is reconstructed from the density histograms using
$\Delta F_{i \to i+1} = -k_BT \ln(f_i / f_{i+1})$
where $f_i$ is the fraction of window $i$'s samples in the overlap region.

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

The `umbrella:` section is added alongside the normal `system:`, `propagate:`,
and `analysis:` sections. The `propagate:` section defines the MC moves used in
both the drive and production phases. The `analysis:` section runs per-window
during production.

```yaml
umbrella:
  cv:
    property: mass_center_separation
    selection: "molecule protein0"
    selection2: "molecule protein1"
  windows:
    range: [25.0, 100.0]
    width: 10.0
    spacing: 6.0
    bin_width: 1.0
  drive:
    force_constant: 5.0
```

### `cv` — collective variable

Any [collective variable](analysis.md#supported-properties) can be used.
Selections are resolved per window.

### `windows` — window layout

| Key         | Required | Default | Description                              |
|-------------|----------|---------|------------------------------------------|
| `range`     | yes      |         | `[min, max]` of the CV range to cover    |
| `width`     | yes      |         | Full width of each window                |
| `spacing`   | yes      |         | Distance between adjacent window centers |
| `bin_width` | no       | `1.0`   | Histogram bin width for the PMF          |

Window centers are placed at $c_i = \text{min} + \text{width}/2 + i \cdot \text{spacing}$.
Each window spans $[c_i - \text{width}/2,\; c_i + \text{width}/2]$.
Adjacent windows must overlap (`spacing < width`) for stitching.

### `drive` — drive phase

| Key              | Required | Description                                  |
|------------------|----------|----------------------------------------------|
| `force_constant` | yes      | Harmonic spring constant $k$ (kJ/mol/unit$^2$) |

During the drive phase a harmonic bias $\frac{1}{2}k(x - c_i)^2$ steers each
walker toward its window center. The drive exits as soon as the CV enters
the window, or errors if `propagate.repeat` sweeps are exhausted.

## Per-window lifecycle

1. **Drive** — harmonic bias pulls the walker into its target window.
   Skipped on restart if a state file already exists.
2. **Production** — hard-wall constraint only. The `propagate:` and `analysis:`
   sections run normally.

## Restart

On restart, existing state files in the `-s` directory are loaded and the
drive phase is skipped. Delete a window's state file to re-drive it.

## Output

- **`pmf.csv`** — three-column CSV: `cv, pmf_kT, stderr_kT`.
  The PMF is shifted so that the minimum is zero.
- **`umbrella_states/window{i}_state.yaml`** — per-window simulation state.
- **`umbrella_states/window{i}_output.yaml`** — per-window energy summary,
  move statistics, and analysis output.
