# MR-GCN Link Prediction Test

This repository contains a small artificial and multimodal dataset to test the feature encoders of the MR-GCN in a link prediction setting. The graph connecting these features has been randomly generated, such that all information must come from the features.

## Getting Started

First, generate the input files:

    python mrgcn/mkdataset.py -c tests/tasks/linkprediction/config.toml -o tests/tasks/linkprediction/ -v

Next, run the MR-GCN with the generated files as input:

    python mrgcn/run.py -c tests/tasks/linkprediction/config.toml -i tests/tasks/linkprediction/LPTEST_<timestamp>.tar -v --dry_run
