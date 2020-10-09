# tflite

A Flutter plugin for accessing TensorFlow Lite API. Supports image classification, object detection ([SSD](https://github.com/tensorflow/models/tree/master/research/object_detection) and [YOLO](https://pjreddie.com/darknet/yolov2/)), [Pix2Pix](https://phillipi.github.io/pix2pix/) and [Deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab) and [PoseNet](https://www.tensorflow.org/lite/models/pose_estimation/overview) on both iOS and Android.

### Table of Contents

- [Installation](#Installation)
- [Usage](#Usage)
    - [Image Classification](#Image-Classification)
    - [Object Detection](#Object-Detection)
      - [SSD MobileNet](#SSD-MobileNet)
      - [YOLO](#Tiny-YOLOv2)
    - [Pix2Pix](#Pix2Pix)
    - [Deeplab](#Deeplab)
    - [PoseNet](#PoseNet)
- [Example](#Example)
    - [Prediction in Static Images](#Prediction-in-Static-Images)
    - [Real-time Detection](#Real-time-Detection)

### Breaking changes

#### Since 1.1.0:

1. iOS TensorFlow Lite library is upgraded from TensorFlowLite　1.x to TensorFlowLiteObjC　2.x. Changes to native code are denoted with `TFLITE2`.

#### Since 1.0.0:

1. Updated to TensorFlow Lite API v1.12.0.
2. No longer accepts parameter `inputSize` and `numChannels`. They will be retrieved from input tensor.
3. `numThreads` is moved to `Tflite.loadModel`.

## Installation

Add `tflite` as a [dependency in your pubspec.yaml file](https://flutter.io/using-packages/).

### Android

In `android/app/build.gradle`, add the following setting in `android` block.

```
    aaptOptions {
        noCompress 'tflite'
        noCompress 'lite'
    }
```

### iOS

Solutions to build errors on iOS:

* 'vector' file not found"

  Open `ios/Runner.xcworkspace` in Xcode, click Runner > Tagets > Runner > Build Settings, search `Compile Sources As`, change the value to `Objective-C++`

* 'tensorflow/lite/kernels/register.h' file not found

  The plugin assumes the tensorflow header files are located in path "tensorflow/lite/kernels".

  However, for early versions of tensorflow the header path is "tensorflow/contrib/lite/kernels".

  Use `CONTRIB_PATH` to toggle the path. Uncomment `//#define CONTRIB_PATH` from here:
  https://github.com/shaqian/flutter_tflite/blob/master/ios/Classes/TflitePlugin.mm#L1

## Usage

1. Create a `assets` folder and place your label file and model file in it. In `pubspec.yaml` add:

```
  assets:
   - assets/labels.txt
   - assets/mobilenet_v1_1.0_224.tflite
```

2. Import the library:

```dart
import 'package:tflite/tflite.dart';
```

3. Load the model and labels:

```dart
String res = await Tflite.loadModel(
  model: "assets/mobilenet_v1_1.0_224.tflite",
  labels: "assets/labels.txt",
  numThreads: 1, // defaults to 1
  isAsset: true, // defaults to true, set to false to load resources outside assets
  useGpuDelegate: false // defaults to false, set to true to use GPU delegate
);
```

4. See the section for the respective model below.

5. Release resources:

```
await Tflite.close();
```

### GPU Delegate

When using GPU delegate, refer to [this step](https://www.tensorflow.org/lite/performance/gpu#step_5_release_mode) for release mode setting to get better performance. 

### Image Classification

- Output format:
```
{
  index: 0,
  label: "person",
  confidence: 0.629
}
```

- Run on image:

```dart
var recognitions = await Tflite.runModelOnImage(
  path: filepath,   // required
  imageMean: 0.0,   // defaults to 117.0
  imageStd: 255.0,  // defaults to 1.0
  numResults: 2,    // defaults to 5
  threshold: 0.2,   // defaults to 0.1
  asynch: true      // defaults to true
);
```

- Run on binary:

```dart
var recognitions = await Tflite.runModelOnBinary(
  binary: imageToByteListFloat32(image, 224, 127.5, 127.5),// required
  numResults: 6,    // defaults to 5
  threshold: 0.05,  // defaults to 0.1
  asynch: true      // defaults to true
);

Uint8List imageToByteListFloat32(
    img.Image image, int inputSize, double mean, double std) {
  var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
  var buffer = Float32List.view(convertedBytes.buffer);
  int pixelIndex = 0;
  for (var i = 0; i < inputSize; i++) {
    for (var j = 0; j < inputSize; j++) {
      var pixel = image.getPixel(j, i);
      buffer[pixelIndex++] = (img.getRed(pixel) - mean) / std;
      buffer[pixelIndex++] = (img.getGreen(pixel) - mean) / std;
      buffer[pixelIndex++] = (img.getBlue(pixel) - mean) / std;
    }
  }
  return convertedBytes.buffer.asUint8List();
}

Uint8List imageToByteListUint8(img.Image image, int inputSize) {
  var convertedBytes = Uint8List(1 * inputSize * inputSize * 3);
  var buffer = Uint8List.view(convertedBytes.buffer);
  int pixelIndex = 0;
  for (var i = 0; i < inputSize; i++) {
    for (var j = 0; j < inputSize; j++) {
      var pixel = image.getPixel(j, i);
      buffer[pixelIndex++] = img.getRed(pixel);
      buffer[pixelIndex++] = img.getGreen(pixel);
      buffer[pixelIndex++] = img.getBlue(pixel);
    }
  }
  return convertedBytes.buffer.asUint8List();
}
```

- Run on image stream (video frame):

> Works with [camera plugin 4.0.0](https://pub.dartlang.org/packages/camera). Video format: (iOS) kCVPixelFormatType_32BGRA, (Android) YUV_420_888.

```dart
var recognitions = await Tflite.runModelOnFrame(
  bytesList: img.planes.map((plane) {return plane.bytes;}).toList(),// required
  imageHeight: img.height,
  imageWidth: img.width,
  imageMean: 127.5,   // defaults to 127.5
  imageStd: 127.5,    // defaults to 127.5
  rotation: 90,       // defaults to 90, Android only
  numResults: 2,      // defaults to 5
  threshold: 0.1,     // defaults to 0.1
  asynch: true        // defaults to true
);
```

### Object Detection

- Output format:

`x, y, w, h` are between [0, 1]. You can scale `x, w` by the width and `y, h` by the height of the image.

```
{
  detectedClass: "hot dog",
  confidenceInClass: 0.123,
  rect: {
    x: 0.15,
    y: 0.33,
    w: 0.80,
    h: 0.27
  }
}
```

#### SSD MobileNet:

- Run on image:

```dart
var recognitions = await Tflite.detectObjectOnImage(
  path: filepath,       // required
  model: "SSDMobileNet",
  imageMean: 127.5,     
  imageStd: 127.5,      
  threshold: 0.4,       // defaults to 0.1
  numResultsPerClass: 2,// defaults to 5
  asynch: true          // defaults to true
);
```

- Run on binary:

```dart
var recognitions = await Tflite.detectObjectOnBinary(
  binary: imageToByteListUint8(resizedImage, 300), // required
  model: "SSDMobileNet",  
  threshold: 0.4,                                  // defaults to 0.1
  numResultsPerClass: 2,                           // defaults to 5
  asynch: true                                     // defaults to true
);
```

- Run on image stream (video frame):

> Works with [camera plugin 4.0.0](https://pub.dartlang.org/packages/camera). Video format: (iOS) kCVPixelFormatType_32BGRA, (Android) YUV_420_888.

```dart
var recognitions = await Tflite.detectObjectOnFrame(
  bytesList: img.planes.map((plane) {return plane.bytes;}).toList(),// required
  model: "SSDMobileNet",  
  imageHeight: img.height,
  imageWidth: img.width,
  imageMean: 127.5,   // defaults to 127.5
  imageStd: 127.5,    // defaults to 127.5
  rotation: 90,       // defaults to 90, Android only
  numResults: 2,      // defaults to 5
  threshold: 0.1,     // defaults to 0.1
  asynch: true        // defaults to true
);
```

#### Tiny YOLOv2:

- Run on image:

```dart
var recognitions = await Tflite.detectObjectOnImage(
  path: filepath,       // required
  model: "YOLO",      
  imageMean: 0.0,       
  imageStd: 255.0,      
  threshold: 0.3,       // defaults to 0.1
  numResultsPerClass: 2,// defaults to 5
  anchors: anchors,     // defaults to [0.57273,0.677385,1.87446,2.06253,3.33843,5.47434,7.88282,3.52778,9.77052,9.16828]
  blockSize: 32,        // defaults to 32
  numBoxesPerBlock: 5,  // defaults to 5
  asynch: true          // defaults to true
);
```

- Run on binary:

```dart
var recognitions = await Tflite.detectObjectOnBinary(
  binary: imageToByteListFloat32(resizedImage, 416, 0.0, 255.0), // required
  model: "YOLO",  
  threshold: 0.3,       // defaults to 0.1
  numResultsPerClass: 2,// defaults to 5
  anchors: anchors,     // defaults to [0.57273,0.677385,1.87446,2.06253,3.33843,5.47434,7.88282,3.52778,9.77052,9.16828]
  blockSize: 32,        // defaults to 32
  numBoxesPerBlock: 5,  // defaults to 5
  asynch: true          // defaults to true
);
```

- Run on image stream (video frame):

> Works with [camera plugin 4.0.0](https://pub.dartlang.org/packages/camera). Video format: (iOS) kCVPixelFormatType_32BGRA, (Android) YUV_420_888.

```dart
var recognitions = await Tflite.detectObjectOnFrame(
  bytesList: img.planes.map((plane) {return plane.bytes;}).toList(),// required
  model: "YOLO",  
  imageHeight: img.height,
  imageWidth: img.width,
  imageMean: 0,         // defaults to 127.5
  imageStd: 255.0,      // defaults to 127.5
  numResults: 2,        // defaults to 5
  threshold: 0.1,       // defaults to 0.1
  numResultsPerClass: 2,// defaults to 5
  anchors: anchors,     // defaults to [0.57273,0.677385,1.87446,2.06253,3.33843,5.47434,7.88282,3.52778,9.77052,9.16828]
  blockSize: 32,        // defaults to 32
  numBoxesPerBlock: 5,  // defaults to 5
  asynch: true          // defaults to true
);
```

### Pix2Pix

> Thanks to [RP](https://github.com/shaqian/flutter_tflite/pull/18) from [Green Appers](https://github.com/GreenAppers)

- Output format:
  
  The output of Pix2Pix inference is Uint8List type. Depending on the `outputType` used, the output is:

  - (if outputType is png) byte array of a png image 

  - (otherwise) byte array of the raw output

- Run on image:

```dart
var result = await runPix2PixOnImage(
  path: filepath,       // required
  imageMean: 0.0,       // defaults to 0.0
  imageStd: 255.0,      // defaults to 255.0
  asynch: true      // defaults to true
);
```

- Run on binary:

```dart
var result = await runPix2PixOnBinary(
  binary: binary,       // required
  asynch: true      // defaults to true
);
```

- Run on image stream (video frame):

```dart
var result = await runPix2PixOnFrame(
  bytesList: img.planes.map((plane) {return plane.bytes;}).toList(),// required
  imageHeight: img.height, // defaults to 1280
  imageWidth: img.width,   // defaults to 720
  imageMean: 127.5,   // defaults to 0.0
  imageStd: 127.5,    // defaults to 255.0
  rotation: 90,       // defaults to 90, Android only
  asynch: true        // defaults to true
);
```

### Deeplab

> Thanks to [RP](https://github.com/shaqian/flutter_tflite/pull/22) from [see--](https://github.com/see--) for Android implementation.

- Output format:
  
  The output of Deeplab inference is Uint8List type. Depending on the `outputType` used, the output is:

  - (if outputType is png) byte array of a png image 

  - (otherwise) byte array of r, g, b, a values of the pixels 

- Run on image:

```dart
var result = await runSegmentationOnImage(
  path: filepath,     // required
  imageMean: 0.0,     // defaults to 0.0
  imageStd: 255.0,    // defaults to 255.0
  labelColors: [...], // defaults to https://github.com/shaqian/flutter_tflite/blob/master/lib/tflite.dart#L219
  outputType: "png",  // defaults to "png"
  asynch: true        // defaults to true
);
```

- Run on binary:

```dart
var result = await runSegmentationOnBinary(
  binary: binary,     // required
  labelColors: [...], // defaults to https://github.com/shaqian/flutter_tflite/blob/master/lib/tflite.dart#L219
  outputType: "png",  // defaults to "png"
  asynch: true        // defaults to true
);
```

- Run on image stream (video frame):

```dart
var result = await runSegmentationOnFrame(
  bytesList: img.planes.map((plane) {return plane.bytes;}).toList(),// required
  imageHeight: img.height, // defaults to 1280
  imageWidth: img.width,   // defaults to 720
  imageMean: 127.5,        // defaults to 0.0
  imageStd: 127.5,         // defaults to 255.0
  rotation: 90,            // defaults to 90, Android only
  labelColors: [...],      // defaults to https://github.com/shaqian/flutter_tflite/blob/master/lib/tflite.dart#L219
  outputType: "png",       // defaults to "png"
  asynch: true             // defaults to true
);
```

### PoseNet

> Model is from [StackOverflow thread](https://stackoverflow.com/a/55288616).

- Output format:

`x, y` are between [0, 1]. You can scale `x` by the width and `y` by the height of the image.

```
[ // array of poses/persons
  { // pose #1
    score: 0.6324902,
    keypoints: {
      0: {
        x: 0.250,
        y: 0.125,
        part: nose,
        score: 0.9971070
      },
      1: {
        x: 0.230,
        y: 0.105,
        part: leftEye,
        score: 0.9978438
      }
      ......
    }
  },
  { // pose #2
    score: 0.32534285,
    keypoints: {
      0: {
        x: 0.402,
        y: 0.538,
        part: nose,
        score: 0.8798978
      },
      1: {
        x: 0.380,
        y: 0.513,
        part: leftEye,
        score: 0.7090239
      }
      ......
    }
  },
  ......
]
```

- Run on image:

```dart
var result = await runPoseNetOnImage(
  path: filepath,     // required
  imageMean: 125.0,   // defaults to 125.0
  imageStd: 125.0,    // defaults to 125.0
  numResults: 2,      // defaults to 5
  threshold: 0.7,     // defaults to 0.5
  nmsRadius: 10,      // defaults to 20
  asynch: true        // defaults to true
);
```

- Run on binary:

```dart
var result = await runPoseNetOnBinary(
  binary: binary,     // required
  numResults: 2,      // defaults to 5
  threshold: 0.7,     // defaults to 0.5
  nmsRadius: 10,      // defaults to 20
  asynch: true        // defaults to true
);
```

- Run on image stream (video frame):

```dart
var result = await runPoseNetOnFrame(
  bytesList: img.planes.map((plane) {return plane.bytes;}).toList(),// required
  imageHeight: img.height, // defaults to 1280
  imageWidth: img.width,   // defaults to 720
  imageMean: 125.0,        // defaults to 125.0
  imageStd: 125.0,         // defaults to 125.0
  rotation: 90,            // defaults to 90, Android only
  numResults: 2,           // defaults to 5
  threshold: 0.7,          // defaults to 0.5
  nmsRadius: 10,           // defaults to 20
  asynch: true             // defaults to true
);
```

## Example

### Prediction in Static Images

  Refer to the [example](https://github.com/shaqian/flutter_tflite/tree/master/example).

### Real-time detection

  Refer to [flutter_realtime_Detection](https://github.com/shaqian/flutter_realtime_detection).

## Run test cases

`flutter test test/tflite_test.dart`