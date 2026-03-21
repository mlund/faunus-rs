# Wang-Landau Flat-Histogram Sampling

Iteratively estimates the density of states _g(CV)_ in collective variable space,
producing a free energy surface _F(CV) = −kT ln g(CV)_ without predefined windows
or force constants. Multiple walkers share a single histogram and bias estimate,
accelerating convergence.

The algorithm follows [Chevallier & Cazals](https://doi.org/10.1016/j.jcp.2020.109366):
exponential reduction of the modification factor until a flatness criterion is met
a configurable number of times, then switching to a 1/t rule for rigorous convergence.

## Usage

```bash
faunus wang-landau -i input.yaml [-s wl_states] [-o free_energy.csv] [-j 4]
```

| Flag | Description | Default |
|------|-------------|---------|
| `-i` | Input YAML file | (required) |
| `-s` | State directory for checkpoints | `wl_states` |
| `-o` | Free energy output file | `free_energy.csv` |
| `-j` | Max parallel walker threads (0 = all cores) | `0` |

## Configuration

The input file must contain a `wang_landau:` section alongside the standard
`system:`, `energy:`, and `propagate:` sections.

```yaml
wang_landau:
  coordinate:
    property: mass_center_separation
    selection:
      molecule: protein
    range: [2.0, 15.0]
    resolution: 0.1
  # coordinate2: ...         # optional, for 2D
  ln_f_initial: 1.0          # initial modification factor
  flatness_threshold: 0.8    # min/mean histogram ratio for flatness
  min_flatness: 20           # flatness checks before switching to 1/t
  min_ln_f: 1.0e-6           # convergence criterion
  steps_per_check: 10000     # MC macro-steps between flatness checks
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `coordinate` | Collective variable (see [analysis](analysis.md)) | (required) |
| `coordinate2` | Second CV for 2D sampling | (none) |
| `ln_f_initial` | Starting value of ln _f_ | 1.0 |
| `flatness_threshold` | Flatness criterion (min/mean of histogram) | 0.8 |
| `min_flatness` | Flatness checks before 1/t transition | 20 |
| `min_ln_f` | Stop when ln _f_ falls below this | 1e-6 |
| `steps_per_check` | MC steps per walker between flatness checks | 10000 |

## Convergence regimes

1. **Exponential**: Each time the histogram is flat, `ln_f → ln_f / 2` and the histogram resets.
2. **1/t**: After `min_flatness` checks, `ln_f = 1/(t+1)` where _t_ is the cumulative update count.

## Restart

The state directory stores per-walker particle states (`walker0_state.yaml`, ...)
and the shared histogram (`histogram.yaml`). Re-running with the same `-s` directory
resumes from the last checkpoint.

## Output

The free energy surface is written to the output file in CSV format.

1D (2 columns):
```
cv,free_energy_kT
2.05,3.421
2.15,2.887
```

2D (3 columns):
```
cv1,cv2,free_energy_kT
2.05,1005.0,3.421
```

## Production run with reweighting

After convergence, the density of states can be used as a static bias to
flatten the free energy surface during a production MC run.
Analysis averages collected under the bias are incorrect;
use [`faunus rerun`](analysis.md#rerun) to replay the trajectory with the
penalty term — reweighting by $w = 1/g(\text{bin})$ is applied automatically.

1. Add the converged checkpoint as a [`penalty`](energy.md#penalty-flat-histogram-bias)
   energy term and run a standard MC simulation with trajectory output.
2. Rerun the trajectory with the same input (including the penalty term)
   to obtain reweighted analysis results.

## Reference

- Chevallier & Cazals, _J. Comput. Phys._ **410**, 109366 (2020).
  [doi:10.1016/j.jcp.2020.109366](https://doi.org/10.1016/j.jcp.2020.109366)
