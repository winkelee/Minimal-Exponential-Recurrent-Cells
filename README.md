  This is the code for the [eMGU and eGRU: Minimal Exponential Recurrent Cells](https://zenodo.org/records/22944853) paper.

  This repository contains detailed notebooks for data preprocessing, training and the evaluation of the models. 

  ## Overview

  We investigate the minimal possible recurrent cells capable of supporting exponential gating as a solution to memory decay. We propose **eMGU** and **eGRU** as the examples of minimal 1-State and 4-State models, remarking that **eMGU** consistently achieves a better performance than MGU, making it an often superior choice in resource-constrained environments. We note a specific failure mode of mLSTM-standalone, the *Autoregressive Attractor Loop* (AAL), and analyze other failure modes of recurrent cells.

  ### Datasets

  We use synthetic and non-synthetic evaluations in the paper. Synthetic evaluations do not require a pre-downloaded dataset, as they are generated proceduraly. 

  Non-Synthetic evaluations require datasets, which we offer to download from here: 

  1. [MNIST](https://systemds.apache.org/datasets/mnist)
  2. [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-v1)

  ### Downloading the pre-trained models

  TBA

  ### Project Structure

  The repository is organized to facilitate the reproduction of the paper's results. 

  `training_and_testing`: Contains Jupyter  notebooks used in training and testing models on respective tests (Synthetic and Non-Synthetic).

  `diagnostics`: Contains Jupyter notebooks used in inner gate diagnostics of all models on all respective tests.

  `helper_scripts`: 
  * `custom_models.py` - Models' definitions for synthetic tasks.
  * `custom_models_for_lm.py` - Models' definitions for language modeling.
  * `custom_models_for_MNIST.py` - Models' definitions for MNIST.
  * `synthetic_testing.py` - The module used throughout the project to support the pipeline with most-used methods.

  `test_figures`: Contains figures with results of synthetic tests.

  `datasets/wikitext2`: Contains the vocabulary extracted from the dataset and used in training and testing all models on language modeling.


  ### Citation

  If you use this code, dataset, or model in your research, please cite our work:

  ```
  @misc{staroverov_2026_22944853,
    author       = {Staroverov, Nikolay},
    title        = {eMGU and eGRU: Minimal Exponential Recurrent Cells},
    month        = sep,
    year         = 2026,
    publisher    = {Zenodo},
    doi          = {10.5281/zenodo.22944853},
    url          = {https://doi.org/10.5281/zenodo.22944853},
  }
  ```

  ### License
  This code is licensed under the MIT License - see the `LICENSE` file for details.
