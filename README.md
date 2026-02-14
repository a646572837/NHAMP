# NHAMP
This package contains models and related scripts to run NHAMP model.
NHAMP is a DGM(deep generating model) for generate AMPs with Non-hemolytic properities.It use ESM2-8M as the decoder and use classifier free guidance transformer to get the sequences.And use attation mask to control variable length peptides generate,it has three guidance on a same target,and you can choose one of them to generate alone,also shown decent results.
# Installation
1.Clone the package.

    git clone https://github.com/a646572837/NHAMP.git
    cd NHAMP

2.Create conda environment using environment file.

    conda env create -f environment.yaml

3.Install NHAMP

    pip install .

# Training Model
All related config is in the file of three notebooks,you can tune them in each single file.
