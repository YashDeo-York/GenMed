# Metrics that Matter: Evaluating Image Quality Metrics for Medical Image Generation

This repository contains the code and resources for the paper: "Metrics that matter: Evaluating image quality metrics for medical image generation."


**Publication:** 

---

**⚠️ This GitHub repository is currently under construction. ⚠️**

The code is being refactored from exploratory notebooks into a more structured and reproducible Python package. We appreciate your patience as we work to make this resource fully available and easy to use.

---

## Abstract

Evaluating generative models for synthetic medical imaging is crucial yet challenging, especially given the high standards of fidelity, anatomical accuracy, and safety required for clinical applications. Standard evaluation of generated images often relies on no-reference image quality metrics (NRIQMs) when ground truth images are unavailable. This study presents a comprehensive assessment of commonly used NRIQMs, investigating their reliability for quantifying the performance of generative models in medical imaging. Using magnetic resonance images of brain tumours and vascular angiography, we systematically evaluate metric sensitivity to various factors including noise, data memorisation, distribution shifts, and localised morphological alterations through a multifaceted evaluation framework comparing upstream metric scores with downstream task-specific performance.

Our findings reveal significant limitations in many widely used no-reference metrics. While sensitive to global distributional characteristics, they often correlate poorly with perceptual quality and downstream task suitability, potentially yielding misleading scores. Furthermore, we identify a profound insensitivity of most tested metrics to localised, clinically significant morphological changes. A major discrepancy emerges between upstream metric assessments and downstream task-specific evaluations, with the latter proving more discriminative and better aligned with practical model utility. We conclude that relying solely on current upstream NRIQMs is insufficient for evaluating generative models in medical imaging and strongly advocate for the integration and prioritisation of downstream task-specific evaluations to ensure clinical relevance and reliability.

## Repository Structure (Planned)

* `src/`: Python package containing modules for data processing, metric calculations, perturbations, downstream evaluation, experiment runners, and plotting.
* `data/`: Instructions on obtaining and setting up the public datasets (IXI, BraTS) used in this study.
* `results_output/`: (Typically .gitignored) Directory where scripts will save generated figures, tables, and metric scores.
* `requirements.txt`: Python dependencies.
* `LICENSE`: Project license.

## Installation & Usage (Forthcoming)

Detailed instructions on setting up the environment, obtaining data, and running the experiments will be provided here once the code refactoring is complete.

## Citation

If you use this work, please cite our paper:

*[Full citation details will be added here upon publication. For now, please refer title above.]*

## Contact

For questions, please contact Yash Deo / yash.deo@york.ac.uk .
