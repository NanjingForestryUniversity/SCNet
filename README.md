# SCNet: A deep learning network framework for analyzing near-infrared spectroscopy using short-cut

## Pre-processing

Since the method we proposed is a regression model, the classification dataset weat kernel is not used in this work.

The other three dataset (corn, marzipan, soil) were preprocessed manually with Matlab and saved in the sub dictionary of `./dataset` dir. The original  dataset of these three dataset were stored in the `./dataset/`. And the data are shared with google drive with this [link](https://drive.google.com/drive/folders/1RFREskNcI2sDv6p7lvLhxFRLUgVTwho6?usp=sharing)

The mango dataset is not in Matlab .m file format, so we save them with the `process.py`.
Meanwhile, we drop the useless part and only save the data between  684 and 900 nm.

> The data set used in this study comprises a total of 11,691 NIR spectra (684–990 nm in 3 nm sampling with a total 103 variables) and DM measurements performed on 4675 mango fruit across 4 harvest seasons 2015, 2016, 2017 and 2018 [24].

The detailed preprocessing progress can be found in [./preprocess.ipynb](./01_preprocess.ipynb)

## Network Training

In order to show our network can prevent degration problem, we hold the experiment which contains the training loss curve of four models. The detailed information can be found in [model_training.ipynb](./02_model_training.ipynb).

The training results were saved on the google drive, here is the [link](https://drive.google.com/drive/folders/1-p1SPg-6lt7i6NkgzUOf5GDhh0cDePsr?usp=sharing])

## Network evaluation

After training our model on training set, we evaluate the models on testing dataset that spared before. The evaluation is done with [model_evaluation.ipynb](03_model_evaluating.ipynb).

