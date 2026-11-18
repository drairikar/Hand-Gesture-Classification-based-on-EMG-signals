# Hand-Gesture Classification from sEMG Signals

Classifying six hand gestures from **8-channel surface EMG** recordings of 36 subjects, comparing classical
machine-learning models with deep networks. Course project for **EE4C12 Machine Learning for Electrical
Engineering**, TU Delft. Joint work by D. Rairikar and S. Patil.

| Class | Gesture |
| ---: | --- |
| 1 | Hand at rest |
| 2 | Hand clenched in a fist |
| 3 | Wrist flexion |
| 4 | Wrist extension |
| 5 | Radial deviation |
| 6 | Ulnar deviation |

Class 0 (unmarked) and class 7 (extended palm, very few samples) are dropped, leaving ~1.5 M balanced samples.

## Notebooks

| Notebook | Contents |
| --- | --- |
| [`EMG_ML_…ipynb`](EMG_ML_S&S_EMG_DRairikar_SPatil.ipynb) | Time-domain features (RMS, simple square integral, absolute differential signal, peak-to-peak), polynomial-kernel SVM, k-NN (k = 4–6) |
| [`EMG_DL_…ipynb`](EMG_DL_S&S_EMG_DRairikar_SPatil.ipynb) | 1-D CNN (2 conv layers + dense) and 2-layer LSTM on the 8-channel samples |

## Results

| Model | Validation accuracy |
| --- | ---: |
| 1-D CNN | ≈ 84% (after ~175 epochs) |
| LSTM | ≈ 82% (after ~40 epochs) |
| SVM / k-NN | Re-run needed, see below |

**Correction to the classical-ML results.** The saved outputs of the ML notebook (k-NN ≈ 99.7%, SVM ≈ 95%)
came from a version in which the `class` column (the target) and the `label` column (the subject id) were
passed to the classifiers as features. That is target leakage, so those scores are not valid. The split now
uses the 8 EMG channels only; re-run the notebook to get the real numbers.

## Limitations and next steps

- The split is random **per sample**, so the same subject and recording appear in both train and test.
  A subject-wise split (`GroupShuffleSplit` on `label`) would measure generalisation to new users.
- Models classify single time samples. Windowed features (e.g. 200 ms windows with RMS, MAV, zero
  crossings) are the standard approach and usually help a lot.
- The DL notebook normalises the test set with its own statistics; use the training mean and std instead.

## Running

```bash
pip install -r requirements.txt
jupyter notebook
```

The course dataset `Project_Data_EE4C12_S&S_EMG.csv` is not redistributed here. The ML notebook reads it
from the notebook folder. The DL notebook was written for Google Colab and reads
`/content/Project_Data_EE4C12_S&S_EMG.csv`; upload the file to Colab, or change that path when running
locally.
