
# Generalized Clustering and Multi-Manifold Learning (GCML)




The code includes the following modules:
* Datasets (MNIST-full, MNIST-test, USPS, Fashion-MNIST, Reuters-10k, HAR, and Pendigits)
* Training for GCML
* Evaluation metrics 
* Visualisation



## Requirements

* pytorch == 1.3.1
* scipy == 1.3.1
* numpy == 1.18.5
* scikit-learn == 0.21.3
* matplotlib == 3.1.1



## Description

* main.py  
  * pretrain() -- Pretraining the model with self-reconstruction Loss
  * train() -- End-to-end training of the GCML model
  * test() -- Test generalization performance on out-of-sample (testing sample)
* autotrain.py -- Scripts for automatic testing on seven datasets
* dataset.py  
  * Dataset() -- Load data of selected dataset
* evaluation.py  
  * GetIndicator() -- Auxiliary tool for evaluating metric 
* loss.py  
  * Loss_calculate() -- Calculate losses: ℒ<sub>LIS</sub>, ℒ<sub>rank</sub>, ℒ<sub>AE</sub>, ℒ<sub>align</sub> 

* model.py  
  * AutoEncoder() -- The architecture used in this work
  * GCML() -- Calculation *Q* distribution and *P* distribution
* utils.py  
  * visualize() -- Auxiliary tools for visualizing intermediate results
  * Clustering() -- For initializing the clustering centers



## Dataset

The datasets used in this paper are available in:

https://drive.google.com/file/d/1nNenJQVBJ-R4B6rs_K_YxGrVyZq4kAfz/view?usp=sharing



## Running the code

1. Install the required dependency packages

2. To get the results on seven datasets, run

  ```
python autotrain.py
  ```

3. To get the metrics and visualisation, refer to

  ```
../plots/dataset/pics/
  ```
where the *dataset* is one of the seven datasets (MNIST-full, MNIST-test, USPS, Fashion-MNIST, Reuters-10k, HAR, and Pendigits)



## Citation

If you find this project useful for your research, please use the following BibTeX entry.

```
@inproceedings{wu2022generalized,
  title={Generalized Clustering and Multi-Manifold Learning with Geometric Structure Preservation},
  author={Wu, Lirong and Liu, Zicheng and Xia, Jun and Zang, Zelin and Li, Siyuan and Li, Stan Z},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={139--147},
  year={2022}
}
```
