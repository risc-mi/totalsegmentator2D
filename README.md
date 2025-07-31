# TotalSegmentator 2D: A Tool for Rapid Anatomical Structure Analysis
> ⚠️ **Update:** The TS2D models [MIUA2025a] have been published with version 1.0.0!

> ⚠️ **Note:** The X-Ray models [MIUA2025b] will be released soon, stay tuned!

> ⚠️ **Note:** This is a preview version and may be incomplete or subject to changes.

## About

**TotalSegmentator 2D (TS2D)** is a tool for fast and efficient anatomical structure segmentation and analysis. **TS2D** is built upon the **TotalSegmentator** dataset and tool, but adopts a 2D projection approach to enable rapid inference and reduced resource consumption. It is designed to segment a variety of anatomical structures in medical images and supports a broad range of applications. The resulting segmentations are used to infer the presence of specific anatomical structures within the image.

TS2D has been used for:
- Anatomical structure segmentation and analysis in CT and X-Ray images.
- Body-region segmentation and detection in CT scans.

<img src="assets/method.png" alt="Overview of our method." style="max-width: 600px; width: 100%;">

_Figure 1: Overview of our method. The volumetric scan ($I_{3D}$) and ground-truth labels ($T_{3D}$) are projected onto the coronal plane to train 2D U-Net model(s). The trained models can then efficiently infer 2D labels ($L_{2D}$) for any projected CT scan._

# Method

TS2D uses coronal projection images generated through maximum and average intensity projection for the segmentation of anatomical structures. The two-channel input is processed by a 2D U-Net, implemented using our [adapted nnU-Net framework](https://github.com/risc-mi/nnUNet-multilabel/blob/main/readme.md) to support multi-label output and thus correctly handle overlapping structures (see Figure 1). The method was trained on the TotalSegmentator v1 dataset, which provides 104 anatomical labels across 1204 CT scans. The task was divided among five models, each trained on a distinct group of anatomical structures (see Figure 2). 


<img src="assets/examples.png" alt="Example segmentation results and DSC scores." style="max-width: 600px; width: 100%;">

_Figure 2: Example segmentation results and DSC scores from each of the five models, with each trained on a specific group of anatomical labels._

We evaluated TS2D against projected ground-truth labels and compared its performance to the original TotalSegmentator tool (TS3D) with inference results projected to 2D.


|   Method    | Overall | Bone Structures | Soft-Tissue Structures | Inference Time (Nvidia RTX 4090) |
|:-----------:|:-------:|:---------------:|:---------------------:|:--------------------------------:|
| TS2D (Ours) | 0.86    | 0.90            | 0.81                  |          0.47–0.86 secs          |
|    TS3D     | 0.97    | 0.97            | 0.97                  |           43–146 secs            |


## Usage

### Setup

TS2D has been tested with **Python 3.12** and Pytorch 2.7.1 (CUDA 11.8) on a **Windows 11** system.
We recommend installing PyTorch in your environment **before installing TS2D**. Ensure you set up PyTorch with the correct CUDA version for your system and PyTorch release. For installation instructions, see the [PyTorch documentation](https://pytorch.org/get-started/locally/).
After setting up PyTorch, install TS2D with pip: `pip install .` on your clone of the repository or run `pip install git+https://github.com/risc-mi/totalsegmentator2D.git`.

### Get Started

You can run TS2D using the Command line interface (CLI):

`python -m ts2d -i <input_image> -o <output_directory>`

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
| TS2D  |   v1    |   ep4000b2    |  cardiac  |  ts2d-v1-ep4000b2_cardiac  |   0.77    |
|       |         |               |  muscles  |  ts2d-v1-ep4000b2_muscles  |   0.93    |
|       |         |               |  organs   |  ts2d-v1-ep4000b2_organs   |   0.78    |
|       |         |               |   ribs    |   ts2d-v1-ep4000b2_ribs    |   0.89    |
|       |         |               | vertebrae | ts2d-v1-ep4000b2_vertebrae |   0.90    |
|       |         |   ep10000b2   |   bones   |  ts2d-v1-ep10000b2_bones   |   0.88    |
|       |         |               |   soft    |   ts2d-v1-ep10000b2_soft   |   0.81    |


Models are specified using a key (e.g., `ts2d`), which can resolve to one or more model IDs (e.g., `ts2d-v1-ep4000b2_organs`).  
A model ID follows the structure `<model>-<dataset>-<configuration>_<group>`. For example, `ts2d-v1-ep4000b2_organs` refers to the TS2D model trained on the TotalSegmentator v1 dataset, with 4000 epochs, batch size 2, for the organ group.  
Model keys can be abbreviated to match multiple models; for instance, `ts2d-v1-ep4000b2` includes all anatomical groups in that configuration. If only `ts2d` is specified, default models are used.

TS2D runs all models matching the specified key and merges their outputs into a single segmentation.  
The default model key is `ts2d-v1-ep4000b2`, which includes the five anatomical group models in this configuration.

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