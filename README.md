# SnailBaLLsp

This repository contains codes of SnailBaLLsp, **a hierarchical attention network with staged curriculum learning for multi‐barcoding‐based Gastropoda identification**. This model is an improvement of [SnailBaLL](https://github.com/BRUCEBINYE/SnailBaLL), and is an end-to-end species identification model for hierarchical taxonomy levels of Gastropoda from subclass to species. More details about SnailBaLLsp could be found in our [paper]().

## Device
we recommend running SnailBaLLsp with a GPU like NVIDIA GeForce RTX 3090, etc. 

## Download
```
git clone https://github.com/BRUCEBINYE/SnailBaLLsp.git
cd ./SnailBallsp
```

## Environment

Firstly, create a new environment `snailballsp` by:

```
conda create --name snailballsp python=3.9
conda activate snailballsp
```

Then you need to install the following packages in the environment

```
conda install pandas=2.2.3 numpy=1.26.3 scikit-learn=1.6.1

pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cu117

pip install mamba-ssm==2.0.3
```


## Embedding extraxtion

SnailBallsp uses RNA secondary structure embeddings of COI, 16S, H3, 18S, ITS1, and ITS2, which are extracted by the RNA foundation model ERNIE-RNA ([https://github.com/Bruce-ywj/ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA)); and also used DNA barcoding embedding of COI extracted by the DNA barcoding foundation model BarcodeMAE ([https://github.com/bioscan-ml/BarcodeMAE](https://github.com/bioscan-ml/BarcodeMAE)). 

Please extract the embeddings of related barcoding types according to the guideline of the two methods. Here, we assume that you have successfully obtained embeddings from DNA barcoding sequences according the manuals of ERNIE-RNA ([https://github.com/Bruce-ywj/ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA)) and BarcodeMAE ([https://github.com/bioscan-ml/BarcodeMAE](https://github.com/bioscan-ml/BarcodeMAE)). 

These embeddings had been deposited in DYRAD [https://doi.org/10.5061/dryad.ttdz08m9c](https://doi.org/10.5061/dryad.ttdz08m9c). You can download embedding files an put them in the folder of `./SnailBaLLsp/` for directly using.

## Training

Training the model with three-stage learning:

```
python run_train.py
```

After saving the trained models, you can apply domain adaptation approach to the model:

```
python run_train_domainAD.py
```

When adding the DNA barcoding embedding of COI as an additional feature of COI to perform training using both RNA secondary structure and DNA barcoding embedding features, you can run:

```
python add_MAE_run_train.py
```

## Independent testing

To predict results of independent testing dataset, particular for the unseen dataset, you can run: 

```
python run_test_independent.py
```

## Citations

If you think SnailBaLLsp is useful, please cite our work when you use it:

Bin Ye, Junfeng Xia, Xia Wan, Min Wu, Satoshi Chiba. [Multi‐barcoding‐based Gastropoda identification using hierarchical attention network with staged curriculum learning](). Submitting.

## License

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
