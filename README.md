# DistractedDriver
Project for ML on the Distracted Driver Kaggle dataset.

## Getting the data

The original datasets can be obtained from the [corresponding Kaggle competition](https://www.kaggle.com/c/state-farm-distracted-driver-detection), by accepting terms and conditions, then downloading, and adding to the `data` subdirectory over the root of this repo to be found by the python scripts.

The original training dataset is then split into Training, Validation and Test sets. Kaggle's test dataset is not labelled and is just used for the competition, therefore it cannot be used for evaluating the model. 

There is a python script to generate the 3 dataset folders [here](python/pre.py). This script makes a random split of 75%, 5% and 20% of the original training data set, into the resulting training, validation and test sets respectively. The split has been done roughly preserving the class distribution for fairness.

Also check some [stats](DistractedDriverCreateML/DistractedDriverCreateML/count_imgs.log) regarding image class distribution and size of the data sets.

In order to be able to compare results precisely, it is possible to recreate the exact data sets used for the experiment from the following lists:
* [Training set](DistractedDriverCreateML/DistractedDriverCreateML/TrainingData.log)
* [Validation set](DistractedDriverCreateML/DistractedDriverCreateML/ValidationData.log)
* [Test set](DistractedDriverCreateML/DistractedDriverCreateML/TestData.log)

## Repo structure
The following projects are included in this repo:
* [DistractedDriverCreateML](DistractedDriverCreateML): a Xcode macOS project with a script to create a model by using the Create ML API in code (i.e. using `MLImageClassifier()`).
* [DistractedDriverCreateMLApp](DistractedDriverCreateMLApp): a Create ML app project for the same data, available in Xcode 11 and macOS Catalina.
* [DistractedDriverInferenceSameTestSetAsCreateML](DistractedDriverInferenceSameTestSetAsCreateML): a Xcode iOS project to test performance of inference on device. This one uses the same test set as the Create ML project. This project may be hard to manage, since it has a long list of files added to the bundle, which slows down Xcode significantly.
* [DistractedDriverInferenceSmallTestSet](DistractedDriverInferenceSmallTestSet): a Xcode iOS project to test performance of inference on device. This one uses a subset of images (which can also be obtained from the [preprocessing python script]((python/pre.py)) with variable `sample` set to `True`). This makes the Xcode project more manageable, since it does not have to include thousands of files in the bundle.
* [python](python): python scripts for 
  * [generating data sets](python/pre.py), 
  * [creating a keras/tensor flow model](python/kaggle_keras.py), (as per the solution by [Robert Ruzzo](https://www.kaggle.com/robruzzo) in Kaggle, subsequently deleted thus no longer publicly available in its original form),
  * plus some other experimental scripts for model conversion and quantization.
