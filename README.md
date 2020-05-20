# VAE Demo

This is a simple project to demostrate how to convert a TF 2.x MNIST conditional VAE model into a TF Lite one then integrate it into an Android application.

## Code Structure

- ml
  - [mnist-classifier.ipynb](https://github.com/jacktseng831/TFLite_VAE_Demo/blob/master/ml/mnist_classifier.ipynb) - MNIST classifier model
  - [cvae.ipynb](https://github.com/jacktseng831/TFLite_VAE_Demo/blob/master/ml/cvae.ipynb) - MNIST conditional VAE model
  - [cvae-loader.ipynb](https://github.com/jacktseng831/TFLite_VAE_Demo/blob/master/ml/cvae-loader.ipynb) - MNIST conditional VAE model analyzer
  - [cvae-tflite-converter.ipynb](https://github.com/jacktseng831/TFLite_VAE_Demo/blob/master/ml/cvae-tflite-converter.ipynb) - TF Lite converter
* android
  * [assets](https://github.com/jacktseng831/TFLite_VAE_Demo/tree/master/android/app/src/main/assets) - where the TF Lite models stored in the Android app project
  * [DigitClassifier.java](https://github.com/jacktseng831/TFLite_VAE_Demo/blob/master/android/app/src/main/java/com/example/vaedemo/DigitClassifier.java) - interpreter for the classifier model
  * [VaeModule.java](https://github.com/jacktseng831/TFLite_VAE_Demo/blob/master/android/app/src/main/java/com/example/vaedemo/VaeModule.java) - interpreter for the VAE model
  * [FullscreenActivity.java](https://github.com/jacktseng831/TFLite_VAE_Demo/blob/master/android/app/src/main/java/com/example/vaedemo/FullscreenActivity.java) - main activity
