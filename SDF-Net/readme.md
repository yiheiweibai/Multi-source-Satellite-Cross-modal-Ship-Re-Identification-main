

# SDF-Net: Structure-aware Disentangled Feature Learning for Optical-SAR Ship Re-identification

> 🤗 **Pretrained Weights:** [https://huggingface.co/Chenfree233/SDF-Net/tree/main](https://huggingface.co/Chenfree233/SDF-Net/tree/main)
>
> 📄 **Paper:** [https://arxiv.org/abs/2603.12588](https://arxiv.org/abs/2603.12588)

Official PyTorch implementation of **SDF-Net**. This project focuses on cross-modal ship re-identification (Re-ID) between Optical and Synthetic Aperture Radar (SAR) imagery by leveraging disentangled feature representation and structural consistency anchors.

## Abstract

> Cross-modal ship re-identification (ReID) between optical and synthetic aperture radar (SAR) imagery is fundamentally challenged by the severe radiometric discrepancy between passive optical imaging and coherent active radar sensing. While existing approaches primarily rely on statistical distribution alignment or semantic matching, they often overlook a critical physical prior: ships are rigid objects whose geometric structures remain stable across sensing modalities, whereas texture appearance is highly modality-dependent. In this work, we propose SDF-Net, a Structure-Aware Disentangled Feature Learning Network that systematically incorporates geometric consistency into optical--SAR ship ReID. Built upon a ViT backbone, SDF-Net introduces a structure consistency constraint that extracts scale-invariant gradient energy statistics from intermediate layers to robustly anchor representations against radiometric variations. At the terminal stage, SDF-Net disentangles the learned representations into modality-invariant identity features and modality-specific characteristics. These decoupled cues are then integrated through a parameter-free additive residual fusion, effectively enhancing discriminative power. Extensive experiments on the HOSS-ReID dataset demonstrate that SDF-Net consistently outperforms existing state-of-the-art methods. The code and trained models are publicly available at https://github.com/cfrfree/SDF-Net.

## Pipeline

![framework](figs/SDF-Net.png)

## HOSS ReID Dataset

Please organize the [HOSS Dataset](https://zenodo.org/records/15860212) following this structure under the `data` directory:

```text
data
├── HOSS
│   ├── bounding_box_test  # Gallery set
│   ├── bounding_box_train # Training set
│   └── query              # Query set
└── OptiSar_Pair           # Pretraining pairs
    ├── 0001
    ├── 0002
    └── ...

```

## Requirements

### Installation

This project requires **Python 3.9+** and **PyTorch 2.2.2+cu118**.

```bash
conda env create -f environment.yml
conda activate sdfnet

```

Key dependencies include `timm`, `yacs`, `opencv-python`, and `Pillow` (for 32-bit SAR image processing).

## Usage

### 1. Pretraining (Optional)

Utilize large-scale Optical-SAR image pairs for cross-modal alignment:

```bash
python train_pair.py --config_file configs/pretrain_transoss.yml

```

### 2. Training (SDF-Net Fine-tuning)

To train the full SDF-Net with **Disentanglement** and **Structural Consistency Loss**:

```bash
python train.py --config_file configs/SDF-Net.yml

```

*Note: Ensure `MODEL.DISENTANGLE` is set to `True` in the config file to enable DFL.*

### 3. Evaluation

```bash
python test.py --config_file configs/SDF-Net.yml TEST.WEIGHT 'logs/SDF-Net/best.pth'

```

## Citation

If you find our work useful for your research, please consider starring this repository and citing our paper:

```bibtex
@article{chen2026sdfnet,
  title={SDF-Net: Structure-Aware Disentangled Feature Learning for Optical-SAR Ship Re-Identification},
  author={Chen, Furui and Wang, Han and Sun, Yuhan and You, Jianing and Lv, Yixuan and Zhou, Zhuang and Tan, Hong and Li, Shengyang},
  journal={arXiv preprint arXiv:2603.12588},
  year={2026}
}
```

## Acknowledgements

This codebase is built upon [TransReID](https://github.com/damo-cv/TransReID) and [TransOSS](https://github.com/Alioth2000/Hoss-ReID). We thank the authors for their excellent work.
