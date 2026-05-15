# NHAMP: Controllable Generation of Non-Hemolytic Antimicrobial Peptides

This repository contains the official implementation and associated scripts for **NHAMP**, a latent diffusion framework designed to generate antimicrobial peptides (AMPs) with explicit non-hemolytic properties.

NHAMP is a Deep Generative Model (DGM) that operates within the continuous embedding space of Protein Language Models (PLMs). It utilizes a frozen ESM-2 encoder/decoder alongside a conditional diffusion transformer. Variable-length peptide generation is efficiently controlled using attention masks. 

To explicitly balance efficacy and safety, the model employs classifier-free guidance. During sampling, guidance can be accumulated from three distinct conditions (Antimicrobial, Non-hemolytic, and Dual-Property) to steer the generative trajectory directly toward the overlap of target activity and safety.
## Installation
1. **Clone the repository:**

   ```bash
   git clone https://github.com/a646572837/NHAMP.git
   cd NHAMP
   
2. **Create the Conda environment:**

   ```bash
   conda env create -f environment.yaml
   conda activate NHAMP
   
3. **Install NHAMP**
   ```bash
   pip install .

## Training Model
All related config is in the file of three notebooks,you can tune them in each single file.

## Usage and Model Training
All core execution scripts are located in the root directory as Jupyter Notebooks. These notebooks are designed to be self-contained, with all hyperparameters and configurations adjustable within the code cells.

You can find the following notebooks:

- diffusion.ipynb: Scripts for training the conditional diffusion model within the ESM-2 latent space.

- decoder.ipynb: Scripts to fine-tune the noise-adapted language model head for accurate sequence decoding.

- sampler.ipynb: Scripts to execute the multi-guided reverse diffusion process and generate novel peptide candidates. Users can toggle different guidance conditions (Antimicrobial, Non-hemolytic, or Both) to explore specific chemical spaces.

To run these, simply navigate to the root folder and launch the Jupyter interface:

   ```bash
   cd NHAMP
   jupyter notebook

## Model Information
This repository contains model from Figshare,https://figshare.com/articles/online_resource/NHAMP/31341757
