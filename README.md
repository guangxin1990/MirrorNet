# MirrorNet: A Feature Alignment-based CNN Model for Discriminating Tuberculosis Screening in CXR Images 

This repositories is used for COVID-19 Classification of CXR Images. At present, it is only in the experimental stage and cannot be used for application!

This is the result of Fold-1 in the cross-validation experiments. The training set, validation set, and test set can be randomly divided into 80%, 10%, and 10% for cross-validation.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Method
<div align="center">
  <img src="./picture/graphic abstract.png" width="600" height="400">
</div>
MirrorNet contains an AlexNet-based backbone and a mirror loss.

## Usage
### 1
Download the  dataset [here](https://drive.google.com/file/d/1f6Gs2SHJxSdZAbprgetLHFBWPdnp-oMf/view?usp=sharing) and put it into `./data`.
### 2
Download the model [here](https://drive.google.com/file/d/1DET5tgMmOQdPOehJR4nvZUJCcEsMIVC-/view?usp=sharing) and put it in to `./result`, and  test with `python test.py`.
### 2
You can also train with `python train.py`, and then test with`python test.py`

## Results
|| Accuracy | 98.62%  ||  Sensitivity | 98.33 %  ||  Specificity  | 98.73 %  ||   PPV   | 96.72 %  ||  NPV  | 99.36 %  ||
