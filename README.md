# Noise Injection via Split Learning

Python research scripts for studying noise injection, retraining, distributed learning, and sensitivity analysis in a split-learning style workflow. The code experiments with adding noise, retraining models, and comparing original, distributed, and ablation-style settings.

## Main Files

```text
split learning.py             # Core split-learning experiment script
split learning2.py            # Alternative or extended split-learning run
distributed.py                # Distributed-learning experiment
retraining.py                 # Retraining workflow
model testing orginal.py      # Original model testing script
split learning敏感度分析.py     # Sensitivity analysis
消融实验.py                    # Ablation experiment
```

## Requirements

Install the scientific Python and deep-learning packages needed by the script you want to run. Typical dependencies may include:

```bash
pip install numpy matplotlib scikit-learn torch torchvision tensorflow
```

Check the imports at the top of each script before running, because the files are independent experiments and may not all use the same framework.

## How to Run

Run a script directly:

```bash
python "split learning.py"
python distributed.py
python retraining.py
```

Use quotes around filenames that contain spaces.

## Notes

This repository is organized as an experimental workspace rather than a packaged library. Some scripts may require local datasets, pretrained models, or path changes before running on a new machine.
