# SnailBaLLsp

This repository contains codes of SnailBaLLsp, **a hierarchical attention network with staged curriculum learning for multi‐barcoding‐based species identification**. This model is an improvement method comparing to the previous [SnailBaLL](https://github.com/BRUCEBINYE/SnailBaLL), and is an end-to-end species identification model for hierarchical taxonomy levels of Gastropoda from subclass to species. More details about SnailBaLLsp could be found in our [paper]().

## Device
We recommend running SnailBaLLsp with a GPU like NVIDIA GeForce RTX 3090, etc. 

## Download
```
git clone https://github.com/BRUCEBINYE/SnailBaLLsp.git
cd SnailBaLLsp
```

## Environment

Firstly, please create a new environment `snailballsp`:

```
conda env create -n snailballsp -f environment.yml
conda activate snailballsp
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.1/mamba_ssm-2.2.1+cu122torch2.3cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

## Embedding extraction

SnailBallsp uses RNA secondary structure embeddings of COI, 16S, H3, 18S, ITS1, and ITS2, which are extracted from the RNA foundation model ERNIE-RNA ([https://github.com/Bruce-ywj/ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA)); and also used DNA barcoding embedding of COI extracted from the DNA barcoding foundation model BarcodeMAE ([https://github.com/bioscan-ml/BarcodeMAE](https://github.com/bioscan-ml/BarcodeMAE)). 

Please extract the embeddings of related barcoding types according to the guideline of the two methods. Here, we assume that you have successfully obtained embeddings of sequences according the manuals of ERNIE-RNA ([https://github.com/Bruce-ywj/ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA)) and BarcodeMAE ([https://github.com/bioscan-ml/BarcodeMAE](https://github.com/bioscan-ml/BarcodeMAE)). 

These embeddings had been deposited in DYRAD [https://doi.org/10.5061/dryad.ttdz08m9c](https://doi.org/10.5061/dryad.ttdz08m9c). You can download embedding files and put them in the folder `./SnailBaLLsp/` for directly using.

## Label

Labels in multiple taxonomy levels for training in our model were also deposited in DYRAD [https://doi.org/10.5061/dryad.ttdz08m9c](https://doi.org/10.5061/dryad.ttdz08m9c). We provided 6 taxonomy levels for every sequence of Gastropoda, including subclass, order, superfamily, family, genus, and species. The number of categories of each taxonomy level is as follows:

Taxonomy level | Number of categories
---- | ----
Subclass | 2
Order | 24
Superfamily | 103
Family | 354
Genus | 2753
Species | 11295

When training your own dataset, you need to edit the number of categories of each taxonomy level in the corresponding codes.

## Training, validation, and testing

Training the model with three-stage learning:

```
python run_train.py
```

After saving the trained models, you can apply domain adaptation approach to the model. 

```
python run_train_domainAD.py
```

When adding the DNA barcoding embedding of COI as an additional feature of COI to perform training using both RNA secondary structure and DNA barcoding embedding features, you can run:

```
python add_MAE_run_train.py
```

We also deposited the models pretrained at every stage in DYRAD [https://doi.org/10.5061/dryad.ttdz08m9c](https://doi.org/10.5061/dryad.ttdz08m9c), and these models could be used directly.

## Independent testing

To predict results of independent testing dataset, particular for the unseen dataset, you can run: 

```
python run_test_independent.py
```

## Citation

If you think SnailBaLLsp is useful, please cite our work when you use it:

B. Y., J. X., X. W., M. W., S. C.. [Multi‐barcoding‐based species identification using hierarchical attention network with staged curriculum learning](). Submitting.

## License

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
