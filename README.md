# tflite

A Flutter plugin for accessing TensorFlow Lite API. Supports Classification and Object Detection on both iOS and Android.

### Table of Contents

- [Installation](#Installation)
- [Usage](#Usage)
    - [Image Classification](#Image%20Classification)
    - [Object Detection](#Object%20Detection)
      - [SSD MobileNet](#SSD%20MobileNet)
      - [YOLO](#Tiny%20YOLOv2)
    - [Pix2Pix](#Pix2Pix)
    - [Deeplab](#Deeplab)
- [Example](#Example)
    - [Prediction in Static Images](#Prediction%20in%20Static%20Images)
    - [Real-time Detection](#Real-time%20Detection)

### Breaking changes since 1.0.0:

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

If you get error like "'vector' file not found", please open `ios/Runner.xcworkspace` in Xcode, click Runner > Tagets > Runner > Build Settings, search `Compile Sources As`, change the value to `Objective-C++`;

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
  numThreads: 1 // defaults to 1
);
```

4. See Image Classication and Object Detection below.

5. Release resources:

```
await Tflite.close();
```

### Image Classification

- Output fomart:
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
  threshold: 0.2    // defaults to 0.1
);
```

- Run on binary:

```dart
var recognitions = await Tflite.runModelOnBinary(
  binary: imageToByteListFloat32(image, 224, 127.5, 127.5),// required
  numResults: 6,    // defaults to 5
  threshold: 0.05,  // defaults to 0.1
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
);
```

### Object Detection

- Output fomart:

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
);
```

- Run on binary:

```dart
var recognitions = await Tflite.detectObjectOnBinary(
  binary: imageToByteListUint8(resizedImage, 300), // required
  model: "SSDMobileNet",  
  threshold: 0.4,                                  // defaults to 0.1
  numResultsPerClass: 2,                           // defaults to 5
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
  anchors: anchors,// defaults to [0.57273,0.677385,1.87446,2.06253,3.33843,5.47434,7.88282,3.52778,9.77052,9.16828]
  blockSize: 32,        // defaults to 32
  numBoxesPerBlock: 5   // defaults to 5
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
  numBoxesPerBlock: 5   // defaults to 5
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
  numBoxesPerBlock: 5   // defaults to 5
);
```

### Pix2Pix

> Thanks to [RP](https://github.com/shaqian/flutter_tflite/pull/18) from [Green Appers](https://github.com/GreenAppers)

- Run on image:

```dart
var result = await runPix2PixOnImage(
  path: filepath,       // required
  imageMean: 0.0,       // defaults to 0.0
  imageStd: 255.0,      // defaults to 255.0
);
```

Output:
```
{
  "filename": outputFile
}
```

- Run on binary:

```dart
var result = await runPix2PixOnBinary(
  binary: binary,       // required;
);
```

Output:
```
{
  "binary": outputBinary
}
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
);
```

Output:
```
{
  "binary": outputBinary
}
```

### Deeplab

> Thanks to [RP](https://github.com/shaqian/flutter_tflite/pull/22) from [see--](https://github.com/see--) for Android implementation.

- Output:
  
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
  outputType: "png"   // defaults to "png"
);
```

- Run on binary:

```dart
var result = await runSegmentationOnBinary(
  binary: binary,     // required;
  labelColors: [...], // defaults to https://github.com/shaqian/flutter_tflite/blob/master/lib/tflite.dart#L219
  outputType: "png"   // defaults to "png"
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
  outputType: "png"        // defaults to "png"
);
```

## Example

### Prediction in Static Images

  Refer to the [example](https://github.com/shaqian/flutter_tflite/tree/master/example).

### Real-time detection

  Refer to [flutter_realtime_Detection](https://github.com/shaqian/flutter_realtime_detection).
