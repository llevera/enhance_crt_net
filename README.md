# Enhance CRT-Net

1. Clone this repo
1. Create a subdirectory for training data
1. Download any of the physionet 2020 training sets, either the whole set from https://physionet.org/content/challenge-2020/1.0.2/ or a subset for example the CSPC_2018 that CRT-Net trained on: https://www.kaggle.com/datasets/bjoernjostein/china-physiological-signal-challenge-in-2018 and extract to the training data directory. The number of files doesn't matter as long as each .hea file has a .mat file
1. Decide to train either to predict all diagnostic classes from the dataset (eg to directly compare to CRT-Net) or on a standardised set, as per PhysioNet challenge. The later uses dx_mapping_scored.csv to limit the set of classes. This is controlled by the 'adjust_classes_for_physionet' parameter
1. Run all cells

## Dependencies to check

-   tensorflow==2.14.1
-   tensorflow-probability==0.22.1
-   keras-nlp==0.11.1
