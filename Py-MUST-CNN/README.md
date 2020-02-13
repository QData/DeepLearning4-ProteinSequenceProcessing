# Pytorch implementation of MUST CNN
Code for the paper "MUST-CNN: A Multilayer  Shift-and-Stitch  Deep  Convolutional Architecture  for Sequence-Based Protein  Structure Prediction" (AAAI 2016)

Zeming Lin, Jack Lanchantin, Yanjun Qi 
University of Virginia



## Data
The data is split up into 2 directories: 4Protein, and cb513. Each directory contains a "data" subdirectory and a "hash" subdirectory. The data subdirectory contains "aa1.dat" which is the raw protein sequence data, as well as each \*tag.dat file which are the class labels for each separate class. The data subdir also contains the psi-blast files. The hash subdirectory contains the dictionary numbers for each of the amino acids and class labels.
The data directories are included in this repository as tar files. Untar the data directory which you choose to use.
```
tar -xvf ./data/4Protein.tar.gz -C ./data/
```

## Usage
To run the small model:
```
python main.py --loaddir ./data --task 4Protein.absolute 4Protein.dssp 4Protein.ssp 4Protein.sa-relative --use_psi_features --kernelsize 9 9 9
```

To run the large model:
```
python main.py --loaddir ./data --task 4Protein.absolute 4Protein.dssp 4Protein.ssp 4Protein.sa-relative --use_psi_features  --hiddenunit=1024 --kernelsize 5 5 3 --input_dropout=0.1 --convdropout=0.3 --save_every_epoch
```
To finetune model on a specific task: use options flag --finetune and --finetune_task to specify task
