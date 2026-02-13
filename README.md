# SnailBaLLsp

This repository contains codes of SnailBaLLsp, **a hierarchical attention network with staged curriculum learning for multi‐barcoding‐based species identification**. This method is an improvement of the previous [SnailBaLL](https://github.com/BRUCEBINYE/SnailBaLL), and has several models for end-to-end species identification to predict hierarchical taxonomy levels of Gastropoda from subclass to species. SnailBaLLsp could also be transfered to other taxa, and users could fine tune the models with their own multi-barcoding data to get specific models suitable to taxa of their interests. More details about SnailBaLLsp could be found in our [paper]().

## Device
We recommend running SnailBaLLsp with a GPU like NVIDIA GeForce RTX 3090, etc. 

## Download
```
git clone https://github.com/BRUCEBINYE/SnailBaLLsp.git
cd SnailBaLLsp
```

## Environment

Firstly, please create a new environment `snailballsp`. Particularly, the `mamba-ssm` module need to be installed individually.

```
conda env create -n snailballsp -f environment.yml
conda activate snailballsp
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.1/mamba_ssm-2.2.1+cu122torch2.3cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

## Embedding extraction

SnailBallsp uses RNA secondary structure embeddings of COI, 16S, H3, 18S, ITS1, and ITS2, which are extracted from the RNA foundation model ERNIE-RNA ([https://github.com/Bruce-ywj/ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA)); and also used DNA barcoding embedding of COI extracted from the DNA barcoding foundation model BarcodeMAE ([https://github.com/bioscan-ml/BarcodeMAE](https://github.com/bioscan-ml/BarcodeMAE)). Generally, the embeddings of related barcoding types could be extracted according to the guideline of the two methods. Here, the processing of embedding extraction is also described for users preparing their own data.

Here, we assume that you have successfully obtained embeddings of sequences from ERNIE-RNA and BarcodeMAE. These embeddings had been deposited in DYRAD [https://doi.org/10.5061/dryad.ttdz08m9c](https://doi.org/10.5061/dryad.ttdz08m9c). You can download embedding files and put them in the folder `./SnailBaLLsp/embedding/` for directly using.



## Model construction (Gastropoda)

### Labels

Labels in multiple taxonomy levels for training in our model has been deposited in DYRAD [https://doi.org/10.5061/dryad.ttdz08m9c](https://doi.org/10.5061/dryad.ttdz08m9c). We provided 6 taxonomy levels for every sequence of Gastropoda, including subclass, order, superfamily, family, genus, and species. The number of categories of each taxonomy level is as follows:

Taxonomy level | Number of categories
---- | ----
Subclass | 2
Order | 24
Superfamily | 103
Family | 354
Genus | 2753
Species | 11295


### Training, validation, and testing

The training, validation, and testing datasets have been deposited in DYRAD [https://doi.org/10.5061/dryad.ttdz08m9c](https://doi.org/10.5061/dryad.ttdz08m9c). You can put these data or data folder in the folder `./SnailBaLLsp/data/` for directly using.

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

And then run domain adaptation approach for this model:

```
python add_MAE_run_train_domainAD.py
```


We have deposited the pretrained models at every stage in DYRAD [https://doi.org/10.5061/dryad.ttdz08m9c](https://doi.org/10.5061/dryad.ttdz08m9c), and these models could be used directly.

### Independent testing

To predict results of independent testing dataset, particular for the unseen dataset, using models trained on the RNA secondary structure embedding features, you can run: 

```
python run_test_independent.py
```

If you want to use models trained on fusion of RNA secondary structure and DNA barcoding embedding features, you can run:

```
python add_MAE_run_test_independent.py
```

## Case study: Prediction (Gastropoda)

The Gastropoda case dataset was obtained from GenBank by downloading sequences released from 2025-06-01 to 2025-10-31. We provided this case study dataset in `./SnailBaLLsp/data/Case_Study_Gastropoda/`.

To predict hierarchical taxonomy of label-unkonwn samples, you can use models trained on the RNA secondary structure embedding features by running:

```
python predict_new_sample.py
```
or use models trained on fusion of RNA secondary structure and DNA barcoding embedding features by running:

```
python add_MAE_predict_new_sample.py
```


## Model transfer: Fine-tune with non-Gastropoda data (Bivalve)

To evaluate the cross-taxon application capability of SnailBaLLsp beyond gastropod groups, we constructed a Bivalvia data as an example of transfer learning data for model transfer. We provided this case study dataset in `./SnailBaLLsp/data/Case_Study_Biv/`. The sample sizes for each Bivalvia group are as follows:

Taxonomy level | Number of categories
---- | ----
Subclass | 3
Order | 17
Superfamily | 33
Family | 71
Genus | 617
Species | 3060

You can conveniently fine-tune SnailBaLLsp using multi-barcode training data by modifying the category numbers at each hierarchical taxonomy levels and some key parameters (e.g., learning rate, epochs):

```
python transfer_casedata.py \
    --data_config ./config/data_config.json \
    --fine_tune_config ./config/fine_tune_config.json \
    --output_dir ./Bivalve_fine_tuned_models \
    --pretrained_model ./pretrained_models/DomainA_BestAcc_after_stage_2_BarcodeMAE.pt \
    --data_task "Bivalve"
```


## Tips

### How to obtain embedding by ERNIE-RNA?

- Build the environment using python 3.9 according to the step "Create Environment with Conda" in the manual of ERNIE-RNA [https://github.com/Bruce-ywj/ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA). 
- Access the pretrained model of ERNIE-RNA according to the step "Access pre-trained models" in its manual. Here we only use the pretrained model for representation, Just to put "ERNIE-RNA_pretrain.pt" in a subfolder named "ERNIE-RNA_checkpoint".
- It better to edit the `extract_embedding.py` in ERNIE-RNA folder in order not to generate "all embedding" and "attention map". To do so, you need to print `#`s at the head of `line 165-171` in the original `extract_embedding.py`. This will significantly reduce the running time and save the disk space for generating only CLS embedding.
- Prepare the input sequence file. You can just list all the sequences in a `txt` file with each sequence in a line and with no header in the text.

- Extract CLS embedding of ERNIE-RNA using the following commands:

```bash
conda activate ERNIE-RNA
cd ./ERNIE-RNA
python extract_embedding.py \
  --seqs_path='./data/Case_Study_Snail/COI_seq_ERNIE-RNA.txt' \
  --save_path='./data/Case_Study_Snail/COI_seq_ERNIE-RNA/' \
  --device=0
```
The output `cls_embedding.npy` will be found in the folder `./data/Case_Study_Snail/COI_seq_ERNIE-RNA/` for downstream using.


### How to obtain embedding by BarcodeMAE?

Please note that BarcodeMAE currently can only generate embeddings for supervised training and test data based on known labels. If you wish to use BarcodeMAE to generate embeddings for samples with unknown labels, please be aware that this may lead to inaccurate predictions in SnailBaLLsp. 

- Follow the BarcodeMAE tutorial ([https://github.com/bioscan-ml/BarcodeMAE](https://github.com/bioscan-ml/BarcodeMAE)) to download the package and set up its environment.
- Construct a dataset that must at least contain the columns `nucleotides`, `order_name`, `<taxa_level>_name`, and `<taxa_level>_index`, where `<taxa_level>` can be any taxonomic level such as class, order, family, genus, species, etc. And `<taxa_level>` should be consistent in `<taxa_level>_name` and `<taxa_level>_index`. The dataset may also include all these columns, separated by commas. Please see details of data format in [https://github.com/bioscan-ml/BarcodeMAE](https://github.com/bioscan-ml/BarcodeMAE).

- When you do not need to distinguish between training and test sets, you can put the two `.csv` files with known labels into the same folder. For example, put `new_COI_seq1.csv` and `new_COI_seq2.csv` together to the folder `new_sample`. And then place this folder in the BarcodeMAE directory.

- Modify `knn_probing.py` provided in the BarcodeMAE directory. 
      - First, add the necessary import statement at line 19:
  ```python
  import numpy as np
  ```
      - Second, at lines 127 and 128, replace "supervised_train.csv" and "unseen.csv" with "new_COI_seq1.csv" and "new_COI_seq2.csv", respectively.
      - Then, after lines 150 and 153, add two lines of code to save the generated embeddings:
  ```python
  np.save('new_COI_seq1_BarcodeMAE_embedding.npy', X_unseen)
  np.save('new_COI_seq2_BarcodeMAE_embedding.npy', X)
  ```
      - Finally, save the modified script as `knn_probing_EDITED.py`.

- Run the folowing commands: 

```bash
cd ./BarcodeMAE
conda activate BarcodeMAE
python barcodebert/knn_probing_EDITED.py \
  --run-name knn_new_sample \
  --data-dir ./new_sample/ \
  --taxon genus \
  --pretrained-checkpoint "./model_checkpoints/best_pretraining.pt"\
  --log-wandb \
  --dataset BIOSCAN-5M
```

Thus, you will obtain new_COI_seq1_BarcodeMAE_embedding.npy and new_COI_seq2_BarcodeMAE_embedding.npy in the BarcodeMAE directory, which are ready for downstream analysis.




## Citation

If you think SnailBaLLsp is useful, please cite our work when you use it:

Bin Ye, Junfeng Xia, Xia Wan, Min Wu, Satoshi Chiba. Multi-barcoding-based species identification using a hierarchical attention network with staged curriculum learning. Methods in Ecology and Evolution, 2026. [https://doi.org/10.1111/2041-210x.70264](https://doi.org/10.1111/2041-210x.70264)

## License

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
