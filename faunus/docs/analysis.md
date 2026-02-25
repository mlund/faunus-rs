# Analysis

Analysis objects sample the system at a given frequency during a simulation
and optionally write results to output files.
They are defined in the `analysis` section of the YAML input.

## Collective Variable

Monitors a collective variable (CV) over the course of a simulation,
recording the instantaneous value and a running average at each sampled step.
This is useful for verifying that a CV remains within the expected range,
for convergence checking, and for post-processing of time/step series data.

If `file` is given, each sampled step writes a line with columns
`step`, `value`, and `running_average`.
The file may be gzip-compressed by using a `.gz` extension.

### Example

```yaml
analysis:
  - !CollectiveVariable
    property: mass_center_position
    selection: "molecule protein"
    dimension: z
    range: [-50.0, 50.0]
    file: cv.dat
    frequency: !Every 100
```

### Options

Key          | Required | Default | Description
------------ | -------- | ------- | -------------------------------------------
`property`   | yes      |         | CV type (see table below)
`range`      | yes      |         | Allowed `[min, max]` interval
`frequency`  | yes      |         | Sample frequency, e.g. `!Every 100`
`dimension`  | no       | `xyz`   | Axis projection (`x`, `y`, `z`, `xy`, …)
`selection`  | depends  |         | Selection expression for one atom or group
`selection2` | depends  |         | Second selection (for two-group properties)
`resolution` | no       |         | Bin width (only used by Penalty)
`file`       | no       |         | Output file path; omit to only track the mean

### Supported properties

Property                 | Selection       | Description
------------------------ | --------------- | -------------------------------------------
`volume`                 | none            | Simulation cell volume
`box_length`             | none            | Cell side length along `dimension`
`atom_position`          | one atom        | Atom position projected onto `dimension`
`size`                   | one group       | Number of active particles in a group
`end_to_end`             | one group       | End-to-end distance of a molecular group
`mass_center_position`   | one group       | Mass center position along `dimension`
`mass_center_separation` | two groups      | Distance between two group mass centers
