# tflite_example

Use tflite plugin to run model on images. The image is captured by camera or selected from gallery (with the help of [image_picker](https://pub.dartlang.org/packages/image_picker) plugin).

## Prerequisites

Dowload [mobilenet_v1_1.0_224.tflite](https://github.com/firebase/quickstart-ios/raw/master/mlmodelinterpreter/MLModelInterpreterExample/Resources/mobilenet_v1_1.0_224.tflite) and place it in ./assets folder.

## Install 

```
flutter packages get
```

## Run

```
flutter run
```

## Caveat

```recognizeImageBinary(image)``` (sample code for ```runModelOnBinary```) is slow on iOS when decoding image due to a [known issue](https://github.com/brendan-duncan/image/issues/55) with image package.
