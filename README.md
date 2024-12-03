# ExoSphere

![Licence](https://img.shields.io/github/license/fuchuanpu/exosphere)
![Last](https://img.shields.io/github/last-commit/fuchuanpu/exosphere)
![Language](https://img.shields.io/github/languages/count/fuchuanpu/exosphere)

## 0x00 Introduction
`Exosphere` is a deep learning based traffic detection system. Particularly, it aims to capture tunneled flooding traffic. 
It leverages deep learning based semantic analysis to extract semantic features, i.e., the features represent strong correlations between flooding packets with similar length patterns.

This repository provides a `PyTorch` based demo, which is easy to reproduce. Meanwhile, we also provide [full version](https://drive.google.com/file/d/1_P8HIs3Q9f_HlA9_x2HMr0q6ScPzrF0g/view?usp=drive_link) of the paper for reference.


## 0x01 Environment

`Exosphere` requires `PyTorch` (CUDA version) for DNN training and testing. It relys on `matplotlib`, `numpy` and `sklearn` for analyzing the detection accuracy.

This demo has been tested on a GPU server with 4 `NVIDIA Tesla V100` (32GB), `Ubuntu` v20.04 official image, `Python` v3.8.10, and `PyTorch` v1.11.0 for `CUDA` v11.3.

| Please be aware that the demo processes four traces simultaneously. Therefore, it is recommended to allocate at least 16MB of memory.
 

## 0x02 Usage
First please download the datasets.
```bash
wget https://www.exosphere.fuchuanpu.xyz/dataset.zip
unzip dataset.zip && rm $_
```

Run the following command to apply `Exosphere` for detecting amplification attack traffic:
```bash
./main.py -c ./config/config_amplification.json
```
The results can be found in `./log/amplification/`. We plot RoC curves in `./figures/amplification/`.

Similarly, we provide examples for detecting other types of DDoS attacks.
```bash
./main.py -c ./config/config_application.json
./main.py -c ./config/config_bruteforce.json
./main.py -c ./config/config_flooding.json
```

Finally, the results can be cleaned by `./clean.sh`.

## 0x03 Maintainer
[Chuanpu Fu](fcp20@tsinghua.edu.cn)

