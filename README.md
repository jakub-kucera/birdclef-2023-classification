# Bird-call classification on BirdCLEF 2023: a classical baseline

I wrote this notebook at the [NTU–IBM Qiskit Hackathon](https://aktualne.cvut.cz/stalo-se/20231016-uspechy-studentu-cvut-na-ibm-qiskit-hackathonu-kvantoveho-pocitani-na-national) in Taipei (August 2023). My team's topic there was quantum machine learning for high-dimensional data: a quantum convolutional neural network in Qiskit, applied to bird-call audio. This repo holds only the classical baseline I built for comparison. The quantum model isn't here.

## Data

The [BirdCLEF 2023](https://www.kaggle.com/competitions/birdclef-2023) training set from Kaggle has 16,941 recordings of 264 African bird species, taken from xeno-canto. The task is to predict each recording's species (`primary_label`). The classes are imbalanced: the most common species has 500 recordings, about 3% of the data.

## Approach

- **Features:** each recording is loaded with librosa. I compute 40 MFCCs (mel-frequency cepstral coefficients) and average them over time, giving one 40-number vector per recording. A 20-coefficient version was also tried. Extracting the features takes about an hour on a laptop.
- **Split:** 90% for training, 10% (1,695 recordings) for testing, `random_state=42`.
- **Models:** a set of scikit-learn classifiers plus CatBoost, mostly with default settings. There is also a grid search over 48 MLP configurations: layer sizes, `alpha` and the learning rate.

## Results

On the held-out 10%:

| Model | Accuracy | Macro precision |
|---|---|---|
| CatBoost | 28.4% | 0.20 |
| Random forest | 24.4% | 0.20 |
| MLP (best of the grid) | 24.2% | – |
| MLP, 3 hidden layers (defaults) | 18.8% | 0.11 |
| k-NN (k = 2) | 12.4% | 0.08 |
| SVC | 11.3% | 0.02 |
| Decision tree | 9.4% | 0.05 |
| AdaBoost | 4.1% | 0.00 |

Averaging the MFCCs over a whole recording throws away how the sound changes over time, so these numbers are only a rough reference point. Competitive BirdCLEF solutions train CNNs on spectrogram windows instead.

## Running it

```bash
pip install numpy pandas librosa scikit-learn catboost xgboost tqdm jupyter
jupyter notebook notebook.ipynb
```

Download the competition data from Kaggle, which needs an account and acceptance of the competition rules. Then set `DATASET_PATH` in the second cell. The notebook can save the extracted features to pickle files, so the slow extraction only has to run once.

## Known limitations

- The notebook is exploratory. It has leftover TODOs and commented-out experiments, and the absolute dataset path is hard-coded.
- Only a single train/test split, with no cross-validation and no class weighting.
- No submission to the competition, which scores 5-second windows of soundscape recordings rather than whole clips.
