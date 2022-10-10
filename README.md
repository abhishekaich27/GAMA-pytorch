## Overview
This repository is a PyTorch implementation of the paper "GAMA: Generative Adversarial Multi-Object Scene Attacks" (NeurIPS'22).

![images](https://github.com/abhishekaich27/abhishekaich27.github.io/blob/66b55a4215b441f785fa5633e909be45343803d3/data/Project_pages/NeurIPS_2022/main_fig.png)

[Project page](https://abhishekaich27.github.io/gama.html)

## Usage
1. Download the two folders from [here](https://drive.google.com/drive/folders/1pmsNESi4ofKGJw19yPZNHeRx9aGxxNUg?usp=sharing) and place them in ```classifer_models``` folders.
2. Install the packages listed in ```requirements.txt```. Creating a conda environment is recommended.
3. To train a perturbation generator, run the following command:
```
python train.py --surr_model_type <surrogate model name> --data_name <voc/coco> --train_dir <path to dataset> --eps <l_infty noise strength> --batch_size 8 --epochs 20 --save_folder v4_vocTrained_models --clip_backbone <clipd model type> | tee <exp name>.txt
```
4. To evaluate a trained perturbation generator, run the following command:
```
python eval.py --data_name <voc/coco> --gen_path <path to trained generator file (.pth)> 
```

## Citing this work
If you find this work is useful in your research, please consider citing:
```
@InProceedings{aich2022gama,
  title={GAMA: Generative Adversarial Multi-Object Scene Attacks},
  author={Aich, Abhishek and Khang-Ta, Calvin and Gupta, Akash and Song, Chengyu and Krishnamurthy, Srikanth V and Asif, M Salman and Roy-Chowdhury, Amit K},
  booktitle = {arXiv preprint arXiv:2209.09502},
  year={2022}
}
```
## Contact
Please contact the first author of this paper - Abhishek Aich (aaich001@ucr.edu) for any further queries.


## Acknowledgement
We thank the authors of the following repositories for making their code open-source.  
1. https://github.com/megvii-research/ML-GCN
2. https://github.com/mingming97/multilabel-cam
3. https://github.com/Alibaba-AAIG/Beyond-ImageNet-Attack
