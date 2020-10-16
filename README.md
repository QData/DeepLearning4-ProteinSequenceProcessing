
# MUST-CNN:	A	Multilayer	Shift-and-Stitch	Deep	Convolutional	Architecture	for	Sequence-Based Protein	Structure	Prediction

Code for the paper "MUST-CNN:	A	Multilayer	Shift-and-Stitch	Deep	Convolutional	Architecture	for	Sequence-Based Protein	Structure	Prediction" (AAAI 2016)

Zeming Lin, Jack Lanchantin, Yanjun Qi <br />
University of Virginia

## [Arxiv Link : MUST-CNN: A Multilayer Shift-and-Stitch Deep Convolutional Architecture for Sequence-Based Protein Structure Prediction](https://arxiv.org/abs/1605.03004)


### Pytorch based code implementation in folder: Py-MUST-CNN

- Old version V1 Implementation: Lua Torch based in folder "old-Torch-code"


## Data
The data is split up into 2 directories: 4Protein, and cb513. Each directory contains a "data" subdirectory and a "hash" subdirectory. The data subdirectory contains "aa1.dat" which is the raw protein sequence data, as well as each *tag.dat file which are the class labels for each separate class. The data subdir also contains the psi-blast files. The hash subdirectory contains the dictionary numbers for each of the amino acids and class labels.




## Citing 


```bibtex

@inproceedings{10.5555/3015812.3015817,
author = {Lin, Zeming and Lanchantin, Jack and Qi, Yanjun},
title = {MUST-CNN: A Multilayer Shift-and-Stitch Deep Convolutional Architecture for Sequence-Based Protein Structure Prediction},
year = {2016},
publisher = {AAAI Press},
abstract = {Predicting protein properties such as solvent accessibility and secondary structure from its primary amino acid sequence is an important task in bioinformatics. Recently, a few deep learning models have surpassed the traditional window based multilayer perceptron. Taking inspiration from the image classification domain we propose a deep convolutional neural network architecture, MUST-CNN, to predict protein properties. This architecture uses a novel multilayer shift-and-stitch (MUST) technique to generate fully dense per-position predictions on protein sequences. Our model is significantly simpler than the state-of-the-art, yet achieves better results. By combining MUST and the efficient convolution operation, we can consider far more parameters while retaining very fast prediction speeds. We beat the state-of-the-art performance on two large protein property prediction datasets.},
booktitle = {Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence},
pages = {27–34},
numpages = {8},
location = {Phoenix, Arizona},
series = {AAAI'16}
}
```

