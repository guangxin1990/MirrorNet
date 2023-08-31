# MirrorNet: A Feature Symmetry-based CNN Model for Discriminating COVID-19 Screening in CXR Images 

This repositories is used for COVID-19 Classification of CXR Images. At present, it is only in the experimental stage and cannot be used for application!

This is the result of Fold-3 in the cross-validation experiments. The training set, validation set, and test set can be randomly divided into 60%, 20%, and 20% for cross-validation.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Usage
### 1
Download the  dataset [here](https://drive.google.com/file/d/1f6Gs2SHJxSdZAbprgetLHFBWPdnp-oMf/view?usp=sharing) and put it into `./data`.
### 2
Download the model [here](https://drive.google.com/file/d/1DET5tgMmOQdPOehJR4nvZUJCcEsMIVC-/view?usp=sharing) and put it in to `./result`, and  test with `python test.py`.
### 2
You can also train with `python train.py`, and then test with`python test.py`

## Results
|| Accuracy | 97.55%      ||  Sensitivity | 98.88 %      ||  Specificity      | 97.05 %  ||

|| AUC      | 99.58%      ||  PPV         | 92.67 %      ||  NPV              | 99.57 %  ||

## License
The dataset is made available for non-commercial purposes only. Any comercial use should get formal permission first.
