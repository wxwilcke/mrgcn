# Relational Graph Convolutional Network for Multimodal Knowledge Graphs

PyTorch implementation of a multimodal relational graph convolutional network (Multimodal R-GCN) for heterogeneous data encoded as knowledge graph, as discussed in our paper [End-to-End Entity Classification on Multimodal Knowledge Graphs](https://arxiv.org/abs/2003.12383) (2020). 

## Getting Started

To install this implementation, clone the repository and run:

```
python setup.py install
```

Once installed, we must first prepare our dataset by running

```
python mrgcn/mkdataset.py --config ./if/<name>.toml --output ./data/ -vv
```

This will create a tar file (`<NAME[unix_time]>.tar`) with all data necessary to run subsequent experiments. To include all supported modalities in the dataset, ensure that `include` is set to `true` in the configuration file for all modalities (we can include/exclude these during training as long as they are included here). The original graph is now no longer needed. Note that we must here choose between letting literal values with the same value become one node (`separate_literals = false`) or keep them as as many nodes as there are literals (`separate_literals = true`). We thus need to create two dataset variations per graph if we want to train on both.

Run the Multimodal R-GCN on the prepared dataset by running:

```
python mrgcn/run.py --input ./data/<NAME[unix_date]>.tar --config ./if/<name>.toml -vv
```

This will report the CE loss and accuracy on the validation set. Use the `--test` flag to report that of the test set.

## Reproduction 

To reproduce our classification experiments we need the configuration files for `AIFB`, `MUTAG`, `BGS`, `AM`, `DMG`, and `SYNTH` as available in the `./if/` folder, and the accompanying graphs which are available [here](https://gitlab.com/wxwilcke/mmkg). Use the version of this repository tagged as [v1.0](https://gitlab.com/wxwilcke/mrgcn/-/tags/v1.0).

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

Note that images are expected to be in binary format and included in the graph. 

## Cite 

While we await our paper to be accepted, please cite us as follows if you use this code in your own research. 

```
@article{wilcke2020mrgcn,
  title={End-to-End Entity Classification on Multimodal Knowledge Graphs},
  author={Wilcke, WX and Bloem, P and de Boer, V and van’t Veer, RH and van Harmelen, FAH},
  year={2020}
}
```
