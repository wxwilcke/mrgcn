# MR-GCN Classification Test

This repository contains a small artificial and multimodal dataset to test the feature encoders of the MR-GCN in a binary classification setting. The graph connecting these features has been randomly generated, such that all information must come from the features.

## Getting Started

First, generate the input files:

    python mrgcn/mkdataset.py -c tests/tasks/classification/config.toml -o tests/tasks/classification/ -v

Next, run the MR-GCN with the generated files as input:

    python mrgcn/run.py -c tests/tasks/classification/config.toml -i tests/tasks/classification/CLTEST_<timestamp>.tar -v --dry_run
