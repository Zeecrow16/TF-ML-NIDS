# TFA-ML Network Intrusion Detection System

This project uses **Time–Frequency Analysis (TFA)** and **binary classification models** to classify network packets as **malicious or benign**. The pipeline includes:

- **Feature extraction** using Continuous Wavelet Transform (CWT) on packet timing features.  
- **Model training** using pre-existing classifiers.  
- **Evaluation** on both benign and simulated attack traffic.  
- **Code quality & testing** following PEP8 standards and using PyUnit. 

## Prerequisites

- Python 3.10+  
- Git (for cloning repo)  

---

## Installation Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/TFA-ML-NIDS.git
cd TFA-ML-NIDS

python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt
