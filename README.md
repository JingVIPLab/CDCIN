# CDCIN for few-shot visual question answering

## Introduction

Cross-modal feature Distribution Calibration Inference Network and few-shot VQA datasets for "Cross-modal Feature Distribution Calibration for Few-shot Visual Question Answering" (AAAI 2024) [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28543)].

We provide the pre-training and training code for  CDCIN  the split files for the FSL COCO-QA, FSL VG-QA, and FSL VQA datasets, located in their respective folders.

If you find our method helpful for your research, please cite:

```
@inproceedings{zhang2024cross,
  title={Cross-Modal Feature Distribution Calibration for Few-Shot Visual Question Answering},
  author={Zhang, Jing and Liu, Xiaoqiang and Chen, Mingzhe and Wang, Zhe},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={7151--7159},
  year={2024}
}
```

Also, if you wish to use our datasets, please consider citing this repo.

## Pre-training for CDCIN

1. Modify your init_weights in`pretrain.py`.

2. Modify your image path and split path in `model\dataloader\fsl_vqa.py`.

3. Run the following code for pre-training

   ```bash
   python pretrain.py
   ```

## Training for CDCIN

1. Modify your pre-trained init_weights in`train_inductive_model.py`.

2. Run the following code for training

   ```bash
   python train_inductive_model.py
   ```
   
