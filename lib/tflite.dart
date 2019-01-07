import 'dart:async';
import 'dart:typed_data';
import 'package:meta/meta.dart';
import 'package:flutter/services.dart';

class Tflite {
  static const MethodChannel _channel = const MethodChannel('tflite');

  static Future<String> loadModel({
    @required String model,
    @required String labels,
  }) async {
    return await _channel.invokeMethod(
      'loadModel',
      {
        "model": model,
        "labels": labels,
      },
    );
  }

  static Future<List> runModelOnImage({
    @required String path,
    int inputSize = 224,
    int numChannels = 3,
    double imageMean = 117.0,
    double imageStd = 1.0,
    int numResults = 5,
    double threshold = 0.1,
    int numThreads = 1,
  }) async {
    return await _channel.invokeMethod(
      'runModelOnImage',
      {
        "path": path,
        "inputSize": inputSize,
        "numChannels": numChannels,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "numResults": numResults,
        "threshold": threshold,
        "numThreads": numThreads,
      },
    );
  }

  static Future<List> runModelOnBinary({
    @required Uint8List binary,
    int numResults = 5,
    double threshold = 0.1,
    int numThreads = 1,
  }) async {
    return await _channel.invokeMethod(
      'runModelOnBinary',
      {
        "binary": binary,
        "numResults": numResults,
        "threshold": threshold,
        "numThreads": numThreads,
      },
    );
  }

  static Future close() async {
    return await _channel.invokeMethod('close');
  }
}
