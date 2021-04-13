## 1.1.2

- Add null safety support for Android

## 1.1.1

- Fix error: ';' expected on Android

## 1.1.0

- Upgrade to TensorFlowLiteObjC 2.2.
- Add support for GPU delegate. 
- Fix label size for YOLO. 

## 1.0.6

* Add support for resources outside packaged assets.
* Upgrade version of Flutter SDK and Android Studio.
* Set mininum SDK version to 2.1.0

## 1.0.5

* Set compileSdkVersion to 28, fixing build error "Execution failed for task ':tflite:verifyReleaseResources'."
* Add notes about CONTRIB_PATH.
* Update pubspec.yaml for Flutter 1.10.0 and later.
* Update example app to use image plugin 2.1.4.

## 1.0.4

* Add PoseNet support

## 1.0.3

* Add an asynch option to offload the TfLite run from the UI thread
* Add Deeplab support

## 1.0.2

* Add pix2pix support
* Make number of detections dynamic in Android

## 1.0.1

* Add detectObjectOnBinary
* Add runModelOnFrame
* Add detectObjectOnFrame

## 1.0.0

* Support Object Detection with SSD MobileNet and Tiny Yolov2.
* Updated to TensorFlow Lite API v1.12.0.
* No longer accepts parameter `inputSize` and `numChannels`. They will be retrieved from input tensor.
* `numThreads` is moved to `Tflite.loadModel`.

## 0.0.5

* Support byte list: runModelOnBinary

## 0.0.4

* Support Swift based project

## 0.0.3

* Pass error message in channel in Android.
* Use non hard coded label size in iOS.

## 0.0.2

* Fixed link.

## 0.0.1

* Initial release.
