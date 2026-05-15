# NHAMP: Controllable Generation of Non-Hemolytic Antimicrobial Peptides

This repository contains the official implementation and associated scripts for **NHAMP**, a latent diffusion framework designed to generate antimicrobial peptides (AMPs) with explicit non-hemolytic properties.

NHAMP is a Deep Generative Model (DGM) that operates within the continuous embedding space of Protein Language Models (PLMs). It utilizes a frozen ESM-2 encoder/decoder alongside a conditional diffusion transformer. Variable-length peptide generation is efficiently controlled using attention masks. 

To explicitly balance efficacy and safety, the model employs classifier-free guidance. During sampling, guidance can be accumulated from three distinct conditions (Antimicrobial, Non-hemolytic, and Dual-Property) to steer the generative trajectory directly toward the overlap of target activity and safety.
## Installation
1. **Clone the repository:**
```bash
   git clone https://github.com/a646572837/NHAMP.git
   cd NHAMP

2. **Create conda environment using environment file：**

    conda env create -f environment.yaml
    conda activate NHAMP

3. **Install NHAMP**

    pip install .

# Training Model
All related config is in the file of three notebooks,you can tune them in each single file.
