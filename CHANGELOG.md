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
