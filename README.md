# GFPGAN Integration for Portrait Image Enhancement

This repository provides a Python function to enhance images using the [GFPGAN](https://github.com/TencentARC/GFPGAN) framework. GFPGAN is a state-of-the-art model for facial image restoration and enhancement.

---

## Installation and Setup

### Prerequisites

Make sure you have the following installed:

- Python (3.9 recommended)
- torch==2.0.1 
- torchvision==0.15.2
- basicsr
- facexlib
- realesrgan

### Clone the GFPGAN Repository

```bash
# Clone the GFPGAN repository as a submodule
git clone https://github.com/TencentARC/GFPGAN.git

# Navigate to the GFPGAN directory
cd GFPGAN

# Install dependencies
pip install -r requirements.txt

# Download the pretrained model (v1.4)
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.4/GFPGANv1.4.pth -P experiments/pretrained_models
```

---

## Structure

The `enhance_with_gfpgan` function allows you to process an input image and generate an enhanced version using the GFPGAN model.

The `main.py` creates a web app on streamlit.


---

## Acknowledgments

This project integrates the GFPGAN framework developed by Tencent ARC Lab. For more details, visit the [GFPGAN repository](https://github.com/TencentARC/GFPGAN).

