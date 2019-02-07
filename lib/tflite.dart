import 'dart:async';
import 'dart:typed_data';
import 'package:meta/meta.dart';
import 'package:flutter/services.dart';

class Tflite {
  static const MethodChannel _channel = const MethodChannel('tflite');

  static Future<String> loadModel({
    @required String model,
    @required String labels,
    int numThreads = 1,
  }) async {
    return await _channel.invokeMethod(
      'loadModel',
      {"model": model, "labels": labels, "numThreads": numThreads},
    );
  }

  static Future<List> runModelOnImage(
      {@required String path,
      double imageMean = 117.0,
      double imageStd = 1.0,
      int numResults = 5,
      double threshold = 0.1}) async {
    return await _channel.invokeMethod(
      'runModelOnImage',
      {
        "path": path,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "numResults": numResults,
        "threshold": threshold
      },
    );
  }

  static Future<List> runModelOnBinary(
      {@required Uint8List binary,
      int numResults = 5,
      double threshold = 0.1}) async {
    return await _channel.invokeMethod(
      'runModelOnBinary',
      {"binary": binary, "numResults": numResults, "threshold": threshold},
    );
  }

  static const anchors = [
    0.57273,
    0.677385,
    1.87446,
    2.06253,
    3.33843,
    5.47434,
    7.88282,
    3.52778,
    9.77052,
    9.16828
  ];

  static Future<List> detectObjectOnImage({
    @required String path,
    String model = "SSDMobileNet",
    double imageMean = 127.5,
    double imageStd = 127.5,
    double threshold = 0.1,
    int numResultsPerClass = 5,
    // Used in YOLO only
    List anchors = anchors,
    int blockSize = 32,
    int numBoxesPerBlock = 5,
  }) async {
    return await _channel.invokeMethod(
      'detectObjectOnImage',
      {
        "path": path,
        "model": model,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "threshold": threshold,
        "numResultsPerClass": numResultsPerClass,
        "anchors": anchors,
        "blockSize": blockSize,
        "numBoxesPerBlock": numBoxesPerBlock
      },
    );
  }

  static Future close() async {
    return await _channel.invokeMethod('close');
  }
}
