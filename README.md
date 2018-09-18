Multimodal Relational Graph Convolutional Network (MRGCN)
=====
[ICT Open 2018 slides](https://www.slideshare.net/XanderWilcke/the-knowledge-graph-for-endtoend-learning-on-heterogeneous-knowledge)

This is a work in progress

Note: this work is based on [1]

# Example

Prepare the AIFB dataset (optional):

```python mkdataset.py -c ../if/aifb.toml -o ../of/ -vv```

Run the MRGCN on the prepared AIFB dataset:

```python mrgcn.py -i ../of/AIFB<unix_date>.tar -c ../if/aifb.toml -o /tmp/ -vv```

## References

[1] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, M. Welling, [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103), 2017
