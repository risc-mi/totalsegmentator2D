# TotalSegmentator 2D: A Tool for Rapid Anatomical Structure Analysis

> ⚡ **Latest Updates**  
> - **2025-08-26**: v1.1.0 released → includes models pretrained on the **TotalSegmentator v2.0.1** 
> - **2025-07-31**: v1.0.0 released → includes **MIUA2025a TS2D models** pretrained on the **TotalSegmentator v1.0** dataset
> - 🚧 **Coming soon**: TSXR — X-Ray segmentation models (see [MIUA2025b])

## What is TS2D?

**TotalSegmentator 2D (TS2D)** is a fast and lightweight tool for anatomical structure segmentation and analysis.<br>
It adapts [TotalSegmentator (3D)](https://github.com/wasserth/TotalSegmentator) by projecting CT scans into **2D views**, enabling:

- ⚡ **Rapid inference** (results in less than a second, compared to several minutes for 3D methods)
- 💻 **Low GPU/CPU requirements**  
- 🧠 **Accurate segmentation** of 117 anatomical structures (see `ts2d-v2` models)
- 🩻 **Segmentation of native 2D X-ray scans** (see the upcoming `tsxr` models)

**Use cases include**:
- Anatomical structure segmentation and analysis in CT images.
- Body-region segmentation and detection in CT scans.
- X-ray analysis (coming soon)  

<img src="assets/method.png" alt="Overview of our method." style="max-width: 800px; width: 100%;">

_Figure 1: Standard CT workflow. Volumetric scans and ground-truth labels are projected onto the coronal plane to train five specialized 2D U-Net models. These models enable fast and efficient inference of 2D anatomical labels for any projected CT scan._

## How It Works

TS2D uses coronal projection images generated through maximum and average intensity projection for the segmentation of anatomical structures. 
The two-channel input is processed by a 2D U-Net, implemented using our [adapted nnU-Net framework](https://github.com/risc-mi/nnUNet-multilabel/blob/main/readme.md) to support multi-label 
output and thus correctly handle overlapping structures (see Figure 1). 
Pretrained models for both version 1 and version 2 of TotalSegmentator are available. Version 2 supports segmentation of 117 anatomical labels.
The segmentation task was distributed across five specialized models, each focused on a distinct group of anatomical structures (see Figure 2).

<img src="assets/examples.png" alt="Example segmentation results." style="max-width: 800px; width: 100%;">

_Figure 2: Segmentation results for the five anatomical group models used with the default TS2D configuration (ts2d-v2-ep4000b2), along with the combined output (Patient s0616).

TS2D was evaluated using projected ground-truth labels and its performance was compared to the original TotalSegmentator tool (TS3D), with both methods' inference results projected to 2D for consistency.
A comprehensive comparison can be found in our publication \[MIUA2025a\].

|   Method    | Overall | Bone Structures | Soft-Tissue Structures | Inference Time (Nvidia RTX 4090) |
|:-----------:|:-------:|:---------------:|:---------------------:|:--------------------------------:|
| TS2D (Ours) | 0.86    | 0.90            | 0.81                  |         **0.5-0.9 secs**         |
|    TS3D     | 0.97    | 0.97            | 0.97                  |           43–146 secs            |
_Note_: The table shows results for the TotalSegmentator v1 dataset to ensure comparability with the original TS3D publication.


## Usage

### Setup

TS2D has been tested with **Python 3.12** and Pytorch 2.7.1 (CUDA 11.8) on a **Windows 11** system and on Ubuntu (tested with CPU only).

👉 Install PyTorch **before** installing TS2D (see [PyTorch setup](https://pytorch.org/get-started/locally/)).

Then install TS2D via:
- from PyPI: `pip install ts2d`
- from a local clone: `pip install .`
- from GitHub: `pip install git+https://github.com/risc-mi/totalsegmentator2D.git`.


### Get Started

You can run TS2D using the Command line interface (CLI):

`ts2d -i <input_image> -o <output_directory>`

or alternatively, you can use the API to run TS2D in your Python scripts:

```
from ts2d import TS2D
with TS2D() as model:
    result = model.predict('<input_image>')
    result.save(dest='<output_directory>')
```

TS2D will project the input image, run the segmentation models and save a multilabel segmentation file to the output directory.
The segmentation labels can be parsed from the metadata, to view the segmentation use e.g. 3D Slicer to view the results.
For more information, refer to the CLI help or the API documentation.

### Model overview

The following models are available in TS2D have been published and can be specified using the `--model` argument in the CLI or the `key` parameter in the API:

| Model | Dataset | Configuration |   Group   |          Model ID          | Test Dice |
|:-----:|:-------:|:-------------:|:---------:|:--------------------------:|:---------:|
| TS2D  | v2.0.1  |   ep4000b2    |  cardiac  |  ts2d-v2-ep4000b2_cardiac  |   0.72    |
|       |         |               |  muscles  |  ts2d-v2-ep4000b2_muscles  |   0.96    |
|       |         |               |  organs   |  ts2d-v2-ep4000b2_organs   |   0.78    |
|       |         |               |   ribs    |   ts2d-v2-ep4000b2_ribs    |   0.88    |
|       |         |               | vertebrae | ts2d-v2-ep4000b2_vertebrae |   0.88    |
|       | v1.0.0  |   ep4000b2    |  cardiac  |  ts2d-v1-ep4000b2_cardiac  |   0.77    |
|       |         |               |  muscles  |  ts2d-v1-ep4000b2_muscles  |   0.93    |
|       |         |               |  organs   |  ts2d-v1-ep4000b2_organs   |   0.78    |
|       |         |               |   ribs    |   ts2d-v1-ep4000b2_ribs    |   0.89    |
|       |         |               | vertebrae | ts2d-v1-ep4000b2_vertebrae |   0.90    |
|       |         |   ep10000b2   |   bones   |  ts2d-v1-ep10000b2_bones   |   0.88    |
|       |         |               |   soft    |   ts2d-v1-ep10000b2_soft   |   0.81    |


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

## Publications

Our following publications are related to the development and application of **TS2D**:

* Original publication introducing TS2D:
  * _[MIUA2025a]_ **TotalSegmentator 2D: A Tool for Rapid Anatomical Structure Analysis**\
  **Presented** at Medical Image Understanding and Analysis (MIUA) Conference 2025\
  Full Reference: `Sabrowsky-Hirsch, B., Alshenoudy, A., Thumfart, S., Giretzlehner, M. (2025).  TotalSegmentator 2D (TS2D): A Tool for Rapid Anatomical Structure Analysis. Medical Image Understanding and Analysis 2025 (MIUA 2025). Springer Nature.`


* TS2D extended to the segmentation of X-Ray images:
  * _[MIUA2025b]_ **Leveraging Synthetic Data for Whole-Body Segmentation in X-ray Images**\
  **Presented** at Medical Image Understanding and Analysis (MIUA) Conference 2025\
   Full Reference: `Alshenoudy, A., Sabrowsky-Hirsch, B., Thumfart, S., Giretzlehner, M. (2025). Leveraging Synthetic Data for Whole-Body Segmentation in X-Ray Images. Medical Image Understanding and Analysis 2025.`


* Our earlier work on body-region segmentation for an industrial usecase:
  * _[AIROV2025]_ **Efficient Automatic Detection of Scanned Body Regions in CT Scans**\
  **Presented** at Austrian Symposium on AI, Robotics, and Vision (AIRoV) Conference 2025\
   Full Reference: `Sabrowsky-Hirsch, Bertram, et al. “Efficient Automatic Detection of Scanned Body Regions in CT Scans.” In Proceedings of the Joint Austrian Computer Vision and Robotics Workshop 2025. Verlag der TU Graz (2025).`


## References

TotalSegmentator 2D builds upon two key works in the field of medical image segmentation:

- **Isensee et al. (2021):** *nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation*. *Nature Methods*, 18, 203–211. [https://doi.org/10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z)

- **Wasserthal et al. (2023):** *TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence*. [https://doi.org/10.1148/ryai.230024](https://doi.org/10.1148/ryai.230024)

## Contact

If you have any inquiries, please open a GitHub issue.

## Acknowledgements

<div style="background-color:white;padding: 1em">
<img src="assets/risc.svg" height="50px"  />
<img src="assets/grants.svg" height="50px"  />
</div>

This project is financed by research subsidies granted by the government of Upper Austria. RISC Software GmbH is Member of UAR (Upper Austrian Research) Innovation Network.

### Versions

- v1.0.0: first release of TS2D including the [MIUA2025a] models.
- v1.1.0: added models trained on the TotalSegmentator v2 dataset