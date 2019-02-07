# tflite

A Flutter plugin for accessing TensorFlow Lite API. Supports Classification and Object Detection on both iOS and Android.

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

- Run the model on image file:

```dart
var recognitions = await Tflite.runModelOnImage(
  path: filepath,   // required
  imageMean: 0.0,   // defaults to 117.0
  imageStd: 255.0,  // defaults to 1.0
  numResults: 2,    // defaults to 5
  threshold: 0.2    // defaults to 0.1
);
```

- Run the model on byte list:

```dart
var recognitions = await Tflite.runModelOnBinary(
  binary: imageToByteList(image, 224, 127.5, 127.5),// required
  numResults: 6,    // defaults to 5
  threshold: 0.05,  // defaults to 0.1
);

Uint8List imageToByteList(Image image, int inputSize, double mean, double std) {
  var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
  var buffer = Float32List.view(convertedBytes.buffer);
  int pixelIndex = 0;
  for (var i = 0; i < inputSize; i++) {
    for (var j = 0; j < inputSize; j++) {
      var pixel = image.getPixel(i, j);
      buffer[pixelIndex++] = (((pixel >> 16) & 0xFF) - mean) / std;
      buffer[pixelIndex++] = (((pixel >> 8) & 0xFF) - mean) / std;
      buffer[pixelIndex++] = (((pixel) & 0xFF) - mean) / std;
    }
  }
  return convertedBytes.buffer.asUint8List();
}

```

- Output fomart:
```
{
  index: 0,
  label: "person",
  confidence: 0.629
}
```

### Object Detection

- SSD MobileNet:

```dart
var recognitions = await Tflite.detectObjectOnImage(
  path: filepath,       // required
  model: "SSDMobileNet" 
  imageMean: 127.5,     
  imageStd: 127.5,      
  threshold: 0.4,       // defaults to 0.1
  numResultsPerClass: 2,// defaults to 5
);
```

- Tiny YOLOv2:

```dart
var recognitions = await Tflite.detectObjectOnImage(
  path: filepath,       // required
  model: "YOLO"         
  imageMean: 0.0,       
  imageStd: 255.0,      
  threshold: 0.3,       // defaults to 0.1
  numResultsPerClass: 2,// defaults to 5
  List anchors: anchors,// defaults to [0.57273,0.677385,1.87446,2.06253,3.33843,5.47434,7.88282,3.52778,9.77052,9.16828]
  blockSize: 32,        // defaults to 32
  numBoxesPerBlock: 5   // defaults to 5
);
```

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

## Demo

Refer to the [example](https://github.com/shaqian/flutter_tflite/tree/master/example).