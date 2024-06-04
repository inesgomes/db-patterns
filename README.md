# Finding Patterns in Ambiguity

![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)

Accepted paper @ [ReGenAI workshop](https://sites.google.com/view/cvpr-responsible-genai/home) - [CVPR 2024](https://cvpr.thecvf.com/)

All experiments were conducted using a machine equipped with a Tesla T4 GPU, ensuring sufficient computational power for training and inference phases.

## Create Virtual Environment

```ssh
conda env create -f environment.yml
```

This environment assumes working with CUDA12.

## Prepare env file

Create .env file with the following information
```yaml
CUDA_VISIBLE_DEVICES=0
FILESDIR=<file directory>
HOMEDIR=<repo directory>
ENTITY=<wandb entity to track experiments>
```
HOMEDIR and FILESDIR are equal if our repository and file directory are the same

**all experiments are saved on [weights&biases](https://wandb.ai/home), so please add your entity**

## Available Datasets

- MNIST
- Fashion MNIST

## Run

### Preparation

1. train binary classifiers for a given subset: `python -m src.classifier --dataset mnist --pos 7 --neg 1 --nf 1,2,4 --epochs 1`

### Step 1: Synthetic Data Generation

In this module, we generate synthetic data close to the decision boundary using GASTeN: variation of GANs that, given a model, generates realistic data that is classiﬁed with low conﬁdence by a given classiﬁer. Results show that the approach can generate images closer to the frontier than the original ones but still realistic. Manual inspection conﬁrms that some of those images are confusing even for humans.

Paper: [GASTeN: Generative Adversarial Stress Test Networks](https://link.springer.com/epdf/10.1007/978-3-031-30047-9_8?sharing_token=XGbq9zmVBDFAEaM4r1AAp_e4RwlQNchNByi7wbcMAY55SAL6inraGCkI72KOuzssTzewKWv51v_1pft7j7WJRbiAzL0vaTmG2vf4gs1QhnZ3lV72H7zSKLWQESXZjq5-1pg77WEnt2EHZaN2b51chvHsO6TW3tiGXSVhUgy87Ts%3D)

**Train GASTeN**:

1. prepare configuration file:
    - go to `experiments/gasten`
        -  this folder contains all GASTeN paper experiments
    - select the experiment
        - e.g. `mnist_7v1.yml`
    - change parameters as needed
    - make sure to update the classifier name in `train[step-2][classifier]`
3. prepare FID score calculation for all pairs of numbers: `python -m src.metrics.fid --dataset mnist --pos 7 --neg 1`
4. create test noise: `python src.utils.gen_test_noise --nz 2048 --z-dim 64`
   - nz minimum value must be 2048 for FID calculation: 
5. train GASTeN: `python -m src --config experiments/original/mnist_7v1.yml`
6. go to wandb and check your experiments

### Step 2, 3 and 4: Finding Patterns in Ambiguity & Prototype Selection and Visualization

This module includes experiments to deep clustering and find prototypes.

1. prepare configuration file:
   - go to `experiments/clustering`
        - this folder contains all clustering + prototypes experiments
   - select the experiment
        - e.g. `mnist_7v1.yml`
   - change parameters as needed, but take into consideration the following:
         - *run_id* is based on previously trained GASTeN. Check run_id in wandb (as job_name).
         - select the GAN epoch that seems to have lower FID and ACD scores
2. run: `python -m src src.clustering --config experiments/patterns/mnist_7v1.yml`
3. go to wandb and check your experiments

## Results 

These results refer to the current experiments found in the experiments folder. Some values may vary, as the parameters are not currently set to be deterministic.

*Datasets and MUT accuracies. The highlighted values represent the highest and lowest MUT accuracy.*

| Dataset       | Subset             | CNN - nf | Accuracy  | Loss |
|---------------|--------------------|----|-----------|------|
| **MNIST**     | *5 vs 3*           | 1  | 92.53%    | 0.23 |
|               |                    | 2  | **92.11%**| 0.29 |
|               |                    | 4  | 96.06%    | 0.10 |
|               | *7 vs 1*           | 1  | 97.50%    | 0.09 |
|               |                    | 2  | 97.41%    | 0.15 |
|               |                    | 4  | 98.84%    | 0.04 |
|               | *8 vs 0*           | 1  | 96.31%    | 0.13 |
|               |                    | 2  | 95.03%    | 0.16 |
|               |                    | 4  | **98.92%**| 0.04 |
| **Fashion-MNIST** | *dress vs top* | 4  | 94.15%    | 0.16 |
|               |                    | 8  | 94.30%    | 0.16 |
|               |                    | 16 | 95.75%    | 0.13 |
|               | *sandal vs sneaker*| 4  | 96.00%    | 0.10 |
|               |                    | 8  | 96.70%    | 0.10 |
|               |                    | 16 | 97.20%    | 0.07 |

*GASTeN results*:

| Dataset        | Subset            | CNN - nf | #test set | #synthetic | FID borderline | FID all |
|----------------|-------------------|-----|-----------|------------|---------------|---------|
| **MNIST**      | *5 vs 3*          | 1   | 169       | 2586       | 278.59        | 23.02   |
|                |                   | 2   | **238**   | **4242**   | **259.83**    | 18.80   |
|                |                   | 4   | 60        | 3495       | 325.33        | 98.09   |
|                | *7 vs 1*          | 1   | 37        | 786        | -             | 21.94   |
|                |                   | 2   | 60        | 969        | -             | 17.94   |
|                |                   | 4   | **11**    | **364**    | -             | 15.43   |
|                | *8 vs 0*          | 1   | 63        | 1605       | -             | 20.84   |
|                |                   | 2   | 75        | 1792       | -             | 20.86   |
|                |                   | 4   | 17        | 673        | -             | **14.38** |
| **Fashion-MNIST** | *dress vs top* | 4   | 75        | 2421       | 334.59        | **109.82** |
|                |                   | 8   | 79        | 2138       | 340.16        | 88.58    |
|                |                   | 16  | 52        | 2024       | **350.81**    | 101.71  |
|                | *sandal vs sneaker*| 4  | 44        | 2904       | 327.78        | 45.88   |
|                |                   | 8   | 32        | 2483       | 338.95        | 48.17   |
|                |                   | 16  | 33        | 2180       | -             | 47.30   |

*Hyperparameters and resulting evaluation metrics for each dataset, where SIL is the silhouette score and DB is the Davies-Bouldin index.*

| Dataset               | CNN | UMAP min_distance | UMAP #components | UMAP #neighbors | GMM #clusters | SIL (↑) | DB (↓) |
|-----------------------|-----|-------------------|------------------|-----------------|---------------|---------|--------|
| *5 vs 3*              | 1   | 0.01              | 60               | 5               | 13            | 0.29    | 1.05   |
|                       | 2   | 0.01              | 60               | 5               | 3             | 0.32    | 1.26   |
|                       | 4   | 0.01              | 60               | 5               | 15            | **0.26**| **1.35**|
| *7 vs 1*              | 1   | 0.01              | 60               | 5               | 3             | 0.47    | 0.76   |
|                       | 2   | 0.01              | 5                | 5               | 15            | 0.39    | 0.94   |
|                       | 4   | 0.02              | 20               | 5               | 3             | **0.52**| **0.70**|
| *8 vs 0*              | 1   | 0.01              | 5                | 5               | 15            | 0.36    | 0.93   |
|                       | 2   | 0.01              | 11               | 5               | 15            | 0.31    | 1.14   |
|                       | 4   | 0.01              | 5                | 5               | 15            | 0.43    | 0.89   |
| *dress vs top*        | 4   | 0.01              | 60               | 5               | 3             | 0.47    | 0.74   |
|                       | 8   | 0.01              | 51               | 7               | 3             | 0.50    | 0.71   |
|                       | 16  | 0.01              | 5                | 5               | 3             | 0.38    | 1.14   |
| *sandal vs sneaker*   | 4   | 0.01              | 5                | 5               | 15            | 0.27    | 1.19   |
|                       | 8   | 0.04              | 5                | 5               | 3             | 0.33    | 0.96   |
|                       | 16  | 0.01              | 60               | 5               | 15            | 0.28    | 1.12   |

