# TRL metrics tracker

This directory contains scripts that automatically update [qgallouedec/trl-metrics](https://huggingface.co/datasets/qgallouedec/trl-metrics), a dataset used to track the evolution of various [TRL](https://github.com/huggingface/trl) metrics.

You can access this dataset using:

```python
>>> from datasets import load_dataset
>>> dataset = load_dataset("qgallouedec/trl-metrics", "pypi_downloads", split="train")
>>> dataset[-1]  # Last entry
{'day': datetime.date(2024, 10, 30), 'num_downloads': 14267}
```

The dataset is updated daily and contains the following metrics:

- `"issue_comments"`: All comments on issues in the repository
- `"issues"`: Issues opened in the repository
- `"models"`: Models uploaded on the Hub
- `"models_likes"`: Likes on models uploaded on the Hub
- `"pypi_downloads"`: Number of downloads from PyPI
- `"stargazers"`: Number of stars on the repository
