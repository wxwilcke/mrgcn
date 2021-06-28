# Relational Graph Convolutional Network for Multimodal Knowledge Graphs

PyTorch implementation of a multimodal relational graph convolutional network (MR-GCN) for heterogeneous data encoded as knowledge graph, as introduced in our paper [End-to-End Learning on Multimodal Knowledge Graphs](http://www.semantic-web-journal.net/content/end-end-learning-multimodal-knowledge-graphs) (2021).

By directly reading N-Triples, a common serialization format for knowledge graphs, the MR-GCN can perform node classification and link prediction on any arbitrary knowledge graph that makes use of the RDF data model. To facilitate multimodal learning, the MR-GCN supports 33 different datatypes encompassing six different modalities, including images, natural language, and spatial information, all of which are automatically inferred from the datatype annotations in the graph and processed accordingly.

## Getting Started

1) To install, clone the repository and run:

```
pip install .
```

Once installed, we must first prepare a dataset by calling `mkdataset` with a configuration file `<dataset>.toml` as argument. For the datasets used in our paper, the configuration files are available in the `./if/` directory. To create a configuration file for a different dataset, simply copy and edit `template.toml`. Note that node classification and link prediction require different options.

2) To prepare a dataset, run

```
python mrgcn/mkdataset.py --config ./if/<dataset>.toml --output ./data/ -vv
```

This will create a tar file (`<DATASET[unix_time]>.tar`) with all data necessary to run subsequent experiments. To include all supported modalities in the dataset, ensure that `include` is set to `true` in the configuration file for all modalities (we can include/exclude these during training as long as they are included here). The original graph is now no longer needed. Note that we must here choose between letting literal values with the same value become one node (`separate_literals = false`) or keep them as many nodes as there are unique literals (`separate_literals = true`). .

3) Run the MR-GCN on the prepared dataset by running:

```
python mrgcn/run.py --input ./data/<DATASET[unix_date]>.tar --config ./if/<dataset>.toml -vv
```

This will report the CE loss and accuracy on the validation set for node classification, and the MRR and hits@k for link prediction. Use the `--test` flag to report that of the test set.

## Reproduction 

To reproduce the experiments of our paper, first acquire the datasets from [here](https://gitlab.com/wxwilcke/mmkg), and use the version of this repository tagged as [v2.0](https://gitlab.com/wxwilcke/mrgcn/-/tags/v2.0). Note that there exists a previous iteration of our paper called [End-to-End Entity Classification on Multimodal Knowledge Graphs](https://arxiv.org/abs/2003.12383) (2020) which only considered node classification and which uses [v1.0](https://gitlab.com/wxwilcke/mrgcn/-/tags/v1.0) of this repository.


## Supported data types

The following data types are supported and automatically encoded if they come with a well-defined data type declaration:

Booleans:

```
- xsd:boolean
```

Numbers:

```
- xsd:decimal
- xsd:double
- xsd:float
- xsd:integer
- xsd:long
- xsd:int
- xsd:short
- xsd:byte

- xsd:nonNegativeInteger
- xsd:nonPositiveInteger
- xsd:negativeInteger
- xsd:positiveInteger

- xsd:unsignedLong
- xsd:unsignedInt
- xsd:unsignedShort
- xsd:unsignedByte
```

Strings:

```
- xsd:string
- xsd:normalizedString
- xsd:token
- xsd:language
- xsd:Name
- xsd:NCName
- xsd:ENTITY
- xsd:ID
- xsd:IDREF
- xsd:NMTOKEN
- xsd:anyURI
```

Time/date:

```
- xsd:date
- xsd:dateTime
- xsd:gYear
```

Spatial:

```
- ogc:wktLiteral
```

Images:

```
- kgbench:base64Image (http://kgbench.info/dt)
```

Note that images are expected to be formatted as binary-encoded strings and included in the graph. 

## Cite 

While we await our paper to be accepted, please cite us as follows if you use this code in your own research. 

```
@article{wilcke2020mrgcn,
  title={End-to-End Learning on Multimodal Knowledge Graphs},
  author={Wilcke, WX and Bloem, P and de Boer, V and van’t Veer, RH},
  year={2021}
}
```
