# tflite

A Flutter plugin for accessing TensorFlow Lite. Supports both iOS and Android.

## Installation

Add `tflite` as a [dependency in your pubspec.yaml file](https://flutter.io/using-packages/).

### Android

In `android/app/build.gradle` file add the following setting in `android` block.

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
);
```

4. Run the model on

- image file:

```dart
var recognitions = await Tflite.runModelOnImage(
  path: filepath,   // required
  inputSize: 224,   // wanted input size, defaults to 224
  numChannels: 3,   // wanted input channels, defaults to 3
  imageMean: 127.5, // defaults to 117.0
  imageStd: 127.5,  // defaults to 1.0
  numResults: 6,    // defaults to 5
  threshold: 0.05,  // defaults to 0.1
  numThreads: 1,    // defaults to 1
);
```

- byte list:

```dart
var recognitions = await Tflite.runModelOnBinary(
  binary: imageToByteList(image, 224, 127.5, 127.5),// required
  numResults: 6,    // defaults to 5
  threshold: 0.05,  // defaults to 0.1
  numThreads: 1,    // defaults to 1
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

5. Release resources:

```
await Tflite.close();
```

## Demo

Refer to the [example](https://github.com/shaqian/flutter_tflite/tree/master/example).