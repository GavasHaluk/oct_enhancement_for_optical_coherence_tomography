# OCT Image Enhancement

Enhancement and quality assessment of Optical Coherence Tomography (OCT) retinal images using classical image processing techniques.

## Overview

This project implements three image enhancement pipelines for OCT images and provides comprehensive evaluation metrics:

| Pipeline | Focus | Techniques |
|----------|-------|------------|
| **Pipeline 1** | Noise Reduction & Contrast | Bilateral filter + CLAHE |
| **Pipeline 2** | Edge Enhancement | Pipeline 1 + Canny edge detection + morphological refinement |
| **Pipeline 3** | Adaptive Processing | Automatic parameter selection based on image quality analysis |

## Features

- **Multiple Enhancement Pipelines**: Compare different processing strategies
- **Comprehensive Metrics**: SSIM, PSNR, Contrast Improvement Index, Edge Preservation
- **Quality-Based Evaluation**: Novel SSIM comparison against high-quality reference images
- **FID Calculation**: Distribution similarity between image sets
- **Batch Processing**: Process entire datasets with progress tracking
- **Visualization**: Automated generation of comparison figures and charts

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/oct-enhancement.git
cd oct-enhancement
pip install -r requirements.txt
```

## Quick Start

### Demo Mode
Process a single image and generate visualizations:
```bash
python main.py --mode demo
```

### Quick Test
Process a sample of images (5 per category):
```bash
python main.py --mode quick --samples 5
```

### Full Processing
Process all images in the dataset:
```bash
python main.py --mode full
```

### Analyze Results
Print detailed analysis of existing results:
```bash
python main.py --mode analyze
```

## Dataset

The project is designed for the [OCT2017 dataset](https://data.mendeley.com/datasets/rscbjbr9sj/3) (Kermany et al., Cell 2018).

Expected directory structure:
```
data_dir/
├── CNV/
├── DME/
├── DRUSEN/
└── NORMAL/
```

Specify a custom data directory:
```bash
python main.py --mode quick --data-dir /path/to/your/data
```

## Output Structure

```
output/
├── pipeline1/          # Enhanced images (noise + contrast)
├── pipeline2/          # Enhanced images (+ edge enhancement)
├── pipeline3/          # Enhanced images (adaptive)
├── metrics/
│   ├── all_metrics.csv # Per-image metrics
│   └── fid_scores.csv  # FID distribution scores
├── figures/            # Charts and plots
└── comparisons/        # Side-by-side comparisons
```

## Metrics

| Metric | Description |
|--------|-------------|
| SSIM | Structural similarity to original (0-1) |
| PSNR | Peak signal-to-noise ratio (dB) |
| Contrast Improvement Index | Ratio of standard deviations (>1 = improvement) |
| Edge Preservation Index | Correlation of edge maps |
| Quality-Based SSIM | Similarity improvement relative to high-quality references |
| FID | Fréchet Inception Distance between image distributions |

## Project Structure

```
oct_enhancement/
├── src/
│   ├── pipeline1.py       # Noise reduction & contrast enhancement
│   ├── pipeline2.py       # Edge detection & boundary enhancement
│   ├── pipeline3.py       # Adaptive comprehensive workflow
│   ├── metrics.py         # Evaluation metrics (SSIM, FID, etc.)
│   ├── batch_processor.py # Batch processing orchestration
│   ├── visualization.py   # Figure generation
│   └── utils.py           # Utility functions
├── main.py                # Main entry point
└── requirements.txt       # Python dependencies
```

## Requirements

- Python 3.8+
- NumPy
- OpenCV
- scikit-image
- Matplotlib
- Pandas
- SciPy
- tqdm

## Results

Processing 1,000 OCT images across four disease categories:

| Metric | Pipeline 1 | Pipeline 2 | Pipeline 3 |
|--------|-----------|-----------|-----------|
| SSIM | 0.573 | 0.545 | 0.438 |
| Contrast Improvement | 1.04× | 1.06× | 1.16× |
| Quality-Based Improvement | 94.9% | 82.1% | 28.4% |

**Key Finding**: Pipeline 1 provides the best balance between enhancement and structure preservation, with 94.9% of images showing improved similarity to high-quality references.

## References

1. Kermany, D.S. et al. "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell*, 2018.
2. Wang, Z. et al. "Image quality assessment: from error visibility to structural similarity." *IEEE TIP*, 2004.
3. Zuiderveld, K. "Contrast Limited Adaptive Histogram Equalization." *Graphics Gems IV*, 1994.

## License

MIT License
