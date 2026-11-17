# Caregiver mirroring and infant facial representations

This repository contains the EEG preprocessing and representational similarity analysis code for the 9- and 30-month cohorts, including notebooks for processing caregiver mirroring codes and constructing facial action-unit models.

## Paper

Bozicevic, L., Massera, A., Murray, L., & Rayson, H. (2026). [Early caregiver mirroring predicts the emergence of action-unit-based facial representations in the infant brain](https://doi.org/10.1016/j.infbeh.2026.102262). *Infant Behavior and Development*, 85, 102262.

## Code

- [`preprocessing_9m/`](preprocessing_9m/) and [`preprocessing_30m/`](preprocessing_30m/) contain MATLAB EEG preprocessing. Each has a `pipeline_01.m` entry point.
- The `face_rsa_9m_*.ipynb` and `face_rsa_30m_*.ipynb` notebooks run cohort-specific RSA. The model RDM, actor, movement-type and shuffled-control analyses are identified in the filenames.
- [`process_2m_coding.ipynb`](process_2m_coding.ipynb) processes the caregiver interaction coding, and [`rdm.py`](rdm.py) and [`rsa.py`](rsa.py) provide model and analysis functions.

Set the data, stimulus and output paths in the scripts and notebooks to your local directories. The repository contains analysis code and supporting files. The original participant EEG and face stimuli are external inputs.

See [LICENSE](LICENSE) for the code licence.
