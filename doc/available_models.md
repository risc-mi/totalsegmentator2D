## Available models

The following models are available in TS2D have been published and can be specified using the `--model` argument in the CLI or the `key` parameter in the API:

| Model | Dataset | Configuration |   Group   |          Model ID          | Test Dice |
|:-----:|:-------:|:-------------:|:---------:|:--------------------------:|:---------:|
| TS2D  | v2.0.1  |   ep4000b2    |  cardiac  |  `ts2d-v2-ep4000b2_cardiac`  |   0.72    |
|       |         |               |  muscles  |  `ts2d-v2-ep4000b2_muscles`  |   0.96    |
|       |         |               |  organs   |  `ts2d-v2-ep4000b2_organs`   |   0.78    |
|       |         |               |   ribs    |   `ts2d-v2-ep4000b2_ribs`    |   0.88    |
|       |         |               | vertebrae | `ts2d-v2-ep4000b2_vertebrae` |   0.88    |
|       | v1.0.0  |   ep4000b2    |  cardiac  |  `ts2d-v1-ep4000b2_cardiac`  |   0.77    |
|       |         |               |  muscles  |  `ts2d-v1-ep4000b2_muscles`  |   0.93    |
|       |         |               |  organs   |  `ts2d-v1-ep4000b2_organs`   |   0.78    |
|       |         |               |   ribs    |   `ts2d-v1-ep4000b2_ribs`    |   0.89    |
|       |         |               | vertebrae | `ts2d-v1-ep4000b2_vertebrae` |   0.90    |
|       |         |   ep10000b2   |   bones   |  `ts2d-v1-ep10000b2_bones`   |   0.88    |
|       |         |               |   soft    |   `ts2d-v1-ep10000b2_soft`   |   0.81    |


### Resolving Model IDs from Keys

Models are specified using a key (e.g., `ts2d`), which can resolve to one or more model IDs (e.g., `ts2d-v1-ep4000b2_organs`).  
A model ID follows the structure `<model>-<dataset>-<configuration>_<group>`. For example, `ts2d-v1-ep4000b2_organs` refers to the TS2D model trained on the TotalSegmentator v1 dataset, with 4000 epochs, batch size 2, for the organ group.  
Model keys can be abbreviated to match multiple models; for instance, `ts2d-v1-ep4000b2` includes all anatomical groups in that configuration. If only `ts2d` is specified, the default models are used (cardiac, muscles, organs, ribs and vertebrae).

TS2D runs all models matching the specified key and merges their outputs into a single segmentation.  
The default model key is `ts2d-v2-ep4000b2`, which includes the five anatomical group models in this configuration.

Example model keys and their resolved model IDs:

<table>
  <thead>
    <tr>
      <th>Key</th>
      <th>Resolved model ID(s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <code>ts2d</code> or<br>
        <code>ts2d-v2</code> or<br>
        <code>ts2d-v2-ep4000b2</code>
      </td>
      <td>
        <code>ts2d-v2-ep4000b2_cardiac</code>,<br>
        <code>ts2d-v2-ep4000b2_muscles</code>,<br>
        <code>ts2d-v2-ep4000b2_organs</code>,<br>
        <code>ts2d-v2-ep4000b2_ribs</code>,<br>
        <code>ts2d-v2-ep4000b2_vertebrae</code></td>
    </tr>
    <tr>
      <td>
<code>ts2d_cardiac</code> or<br>
<code>ts2d-v2_cardiac</code>
</td>
      <td><code>ts2d-v2-ep4000b2_cardiac</code>
</td>
    </tr>
    <tr>
      <td><code>ts2d-v1</code></td>
      <td>
        <code>ts2d-v1-ep4000b2_cardiac</code>,<br>
        <code>ts2d-v1-ep4000b2_muscles</code>,<br>
        <code>ts2d-v1-ep4000b2_organs</code>,<br>
        <code>ts2d-v1-ep4000b2_ribs</code>,<br>
        <code>ts2d-v1-ep4000b2_vertebrae</code>
</td>
    </tr>
    <tr>
      <td><code>ts2d_bones</code></td>
      <td><code>ts2d-v1-ep10000b2_bones</code></td>
    </tr>
  </tbody>
</table>