# Data for "Predicting thermal properties of crystals using machine learning"

This project enables the readers of our paper to reproduce our machine learning (ML) results. Below is a summary of the content of the project, as well as the procedure to run the *ApplyModels_Phonons.py* script.

## Description of content
The folder contains the following files:
- ApplyModels_Phonons.py: the main python script to run the ML models on the row training and test data
- X_scalar_id.csv: The full descriptor sets *before* dimensionality reduction, along with the MaterialsProject ID for each material
- C_v/: The data files and folders for the specific heat capacity ML calculations
  - Figs/: Where the figures will be kept
  - SavedModels/: Where the ML models are saved, and will be loaded
  - SplitDataSets/: Where the 80/20 split training and test sets are saved, and will be loaded
  - X.csv: The CSV file with the dimensionality-reduced descriptors *prior* to train/test splitting
  - y_C_v.csv: The CSV file that has the values of C_v corresponding to the materials in X.csv *prior* to splitting
- entropy/: The data files and folders for the entropy ML calculations. It has the same folder structure as C_v
- eps_total_effective/: The data files and folders for the effective polycrystalline dielectric function ML calculations. It has the same folder structure as C_v

## Procedure for running the ML code

1. Download the whole content of the folder into a directory, let's call it ML
2. Unzip the three folders C_v.zip, entropy.zip and eps_total_effective.zip
3. Start a shell and cd to the folder ML
4. Type the command python ApplyModels_Phonons.py on your shell

For any questions, you can reach me at sherif@sheriftawfikabbas.com or sherif.abbas@rmit.edu.au
