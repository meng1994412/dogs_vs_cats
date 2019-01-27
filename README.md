# Dogs vs. Cats
## Objectives
Implement AlexNet from scratch and transfer learning via ResNet to obtain >90% accuracy on detecting whether input image is a dog or cat:
* Built raw images into `HDF5` dataset suitable for training a deep neural network.
* Built mean subtraction, patch, and over-sampling pre-processors designed to increase the classification accuracy.
* Defined a `HDF5` dataset generator class responsible for yielding batches of images and labels from `HDF5` dataset.  
* Constructed AlexNet from scratch.
* Trained and evaluated AlexNet by using the training, validation, and testing sets.
* Extracted features by using ResNet.
* Trained and evaluated a logistic regression classifier by using the extracted features from previous step.

## Packages Used
* Python 3.6
* [OpenCV](https://docs.opencv.org/3.4.4/) 3.4.4
* [keras](https://keras.io/) 2.2.4
* [Tensorflow](https://www.tensorflow.org/install/) 1.12.0
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) 9.0
* [cuDNN](https://developer.nvidia.com/cudnn) 7.1.2
* [scikit-learn](https://scikit-learn.org/stable/) 0.20.2
* [Imutils](https://github.com/jrosebr1/imutils)
* [NumPy](http://www.numpy.org/)

## Approaches
The dataset is from [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data), which contains 25,000 color images of dogs and cats.

Figure 1 shows examples of the cat and the dog.

<img src="https://github.com/meng1994412/dogs_vs_cats/blob/master/output/cat_example.jpg" height="200">
<img src="https://github.com/meng1994412/dogs_vs_cats/blob/master/output/dog_example.jpg" height="200">

Figure 1: Cat (left) and Dog (right) examples from dataset.


### Build `HDF5` dataset
The `dogs_vs_cats_config.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/config/dogs_vs_cats_config.py)) under `config/` directory stores all relevant configurations for the project, including the paths to input images, total number of class labels, information on the training, validation, and testing splits, path to the HDF5 datasets, and path to output models, plots, and etc.

The `hdf5datasetwriter.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/pipeline/io/hdf5datasetwriter.py)) under `pipeline/io/` directory, defines a class that help to write raw images or features into `HDF5` dataset.

The `build_dogs_vs_cats.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/build_dogs_vs_cats.py)) is used for serializing the raw images into an `HDF5` dataset. Although `Keras` has methods that can allow us to use the raw file paths on disk as input to the training process, this method is highly inefficient. Each and every image residing on disk requires an I/O operation which introduces latency into training pipeline. Not only is `HDF5` capable of storing massive dataset, but it is optimized for I/O operations.

**Caution**: after running the following command, we will have `train.hdf5` about 30G, `val.hdf5` and `test.hdf5` each about 4G.
```
python build_dogs_vs_cats.py
```

### Build image pre-processors
The `meanpreprocessor.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/pipeline/preprocessing/meanpreprocessor.py)) under `pipeline/preprocessing/` directory subtracts the mean red, green, and blue pixel intensties across the training set, which is a form of data normalization. Mean subtraction is used to reduce the affects of lighting variations during classification.

The `dogs_vs_cats_mean.json` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/output/dogs_vs_cats_mean.json)) under `output/` directory will have mean red, green, and blue pixel intensities across the training set when we build the `HDF5` dataset.

The `patchpreprocessor.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/pipeline/preprocessing/patchpreprocessor.py)) under `pipeline/preprocessing/` directory is used to randomly extract MxN pixel regions from an image during training (another type of data augmentation). We apply patch preprocessing when the spatial dimensions of our input images are larger than what the CNN architecture expects. Such preprocessor can help to avoid overfitting.

The `croppreprocessor.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/pipeline/preprocessing/croppreprocessor.py)) under `pipeline/preprocessing/` directory used at testing time to sample five regions of an input image (four corners + center area) along with their corresponding horizontal flips (for total 10 crops). These 10 samples will be passed through the CNN and then the probabilities averaged. Apply this over-sampling methods tends to increase the classification about 1-2 percent.

These three pre-processors can help to increase the classification accuracy.

There are some other pre-processors including:
The `simplepreprocessor.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/pipeline/preprocessing/simplepreprocessor.py)) under `pipeline/preprocessing/` directory defines a class to change the size of image.

The `aspectawarepreprocessor.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/pipeline/preprocessing/aspectawarepreprocessor.py)) under `pipeline/preprocessing/` directory defines a class to change the size of image with respect to aspect ratio of the image.

The `imagetoarraypreprocessor.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/pipeline/preprocessing/imagetoarraypreprocessor.py)) under `pipeline/preprocessing/` directory defines a class to convert the image dataset into keras-compatile arrays.

### Build `HDF5` dataset generator
The `hdf5datasetgenerator.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/pipeline/io/hdf5datasetgenerator.py)) under `pipeline/io/` directory yields batches of images and labels from `HDF5` dataset. This class can help to facilitate our ability to work with datasets that are too big to fit into memory.

### Construct AlexNet from scratch
The AlexNet architecture can be found in `alexnet.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/pipeline/nn/conv/alexnet.py)) under `pipeline/nn/conv/` directory. The input to the model includes dimensions of the image (height, width, depth, and number of classes). In this project, the input would be (width = 227, height = 227, depth = 3, classes = 2). And L2 regularization is used (default value 0.0002).

Table 1 shows the architecture of AlexNet. The activation and batch normalization layer is not shown in the table, which should be after each `CONV` layer and `FC` layer. The `ReLU` activation function is used in the project.

| Layer Type    | Output Size     | Filter Size / Stride    |
| ------------- |:---------------:| -----------------------:|
| Input Image   | 227 x 227 x 3   |                         |
| CONV          | 55 x 55 x 96    | 11 x 11 / 4 x 4, K = 96 |
| POOL          | 27 x 27 x 96    | 3 x 3 / 2 x 2           |
| DROPOUT       | 27 x 27 x 96    |                         |
| CONV          | 27 x 27 x 256   | 5 x 5, K = 256          |
| POOL          | 13 x 13 x 256   | 3 x 3 / 2 x 2           |
| DROPOUT       | 13 x 13 x 256   |                         |
| CONV          | 13 x 13 x 384   | 3 x 3, K = 384          |
| CONV          | 13 x 13 x 384   | 3 x 3, K = 384          |
| CONV          | 13 x 13 x 384   | 3 x 3, K = 256          |
| POOL          | 13 x 13 x 256   | 3 x 3 / 2 x 2           |
| DROPOUT       | 6 x 6 x 256     |                         |
| FC            | 4096            |                         |
| DROPOUT       | 4096            |                         |
| FC            | 4096            |                         |
| DROPOUT       | 4096            |                         |
| FC            | 1000            |                         |
| softmax       | 1000            |                         |
Table 1: Architecture of AlexNet.

### Train and evaluate the AlexNet
The `train_alexnet.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/pipeline/nn/conv/alexnet.py)) is responsible for training the model, plotting the training loss and accuracy curve (for both training and validation sets), and serializing the model to disk.

The `trainingmonitor.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/pipeline/callbacks/trainingmonitor.py)) under `pipeline/callbacks/` directory create a `TrainingMonitor` callback that will be called at the end of every epoch when training a network. The monitor will construct a plot of training loss and accuracy. Applying such callback during training will enable us to babysit the training process and spot overfitting early, allowing us to abort the experiment and continue trying to tune parameters.

We can use the following command to train the network.
```
python train_alexnet.py
```

The `crop_accuracy.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/crop_accuracy.py)) evaluates the model by using the testing set.

The `ranked.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/pipeline/utils/ranked.py)) under `pipeline/utils/` directory contains a helper function to measure both the rank-1 and rank-5 accuracy when the model is evaluated by using testing set.

We can use the following command to evaluate the network.
```
python crop_accuracy.py
```

### Extract features via ResNet (transfer learning)
The `extract_features.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/extract_features.py)) implements transfer learning, which extracts features via `ResNet50`.

We can use the following command to extract features from dataset.
```
python extract_features.py --dataset dataset/kaggle_dogs_vs_cats/train --output output datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5
```

### Train and evaluate the Logistic Regression classifier
The `train_model.py` ([check here](https://github.com/meng1994412/dogs_vs_cats/blob/master/train_model.py)) trains a logistic regression classifier by using the features extracted from previous step, uses grid search to find optimal `C` value, and evaluates the classifier, and finally serialize the model to disk. The grid search could take a quite long time and large memory to proceed, which might not work in any situation. If grid search does not work, then we could just use Logistic Regression.

We can use the following command to train and evaluate logistic regression classifier.
```
python train_model.py --db datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5 --model dogs_vs_cats.cpickle
```

## Results
### Train and evalutate the AlexNet
Figure 2 shows the plot of training loss and accuracy for 75 epochs. Figure 3 demonstrates the rank-1 accuracy without/with using over-sampling pre-processor. With using over-sampling in testing set, we can boost accuracy about 1%, from 92.55% to 93.76%.

<img src="https://github.com/meng1994412/dogs_vs_cats/blob/master/output/3687.png" width="600">

Figure 2: Plot of training loss and accuracy (training + validation sets).


<img src="https://github.com/meng1994412/dogs_vs_cats/blob/master/output/crop_accuracy.png" width="350">

Figure 3: rank-1 accuracy without/with using over-sampling pre-processor.

### Train and evaluate the Logistic Regression classifier from extracted features.
Figure 4 illustrates the evaluation of the Logistic Regression. By using transfer learning technique (feature extraction via `ResNet50`), we can increase the accuracy up to 98.93%.

<img src="https://github.com/meng1994412/dogs_vs_cats/blob/master/output/transfer_learning.png" width="400">

Figure 4: Evaluation of Logistic Regression classifier.
