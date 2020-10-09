import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tflite/tflite.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const MethodChannel channel = MethodChannel(
    'tflite',
  );

  final List<MethodCall> log = <MethodCall>[];

  setUp(() async {
    channel.setMockMethodCallHandler((MethodCall methodCall) {
      log.add(methodCall);
      return null;
    });
    log.clear();
  });
  test('loadModel', () async {
    await Tflite.loadModel(
      model: 'assets/mobilenet_v1_1.0_224.tflite',
      labels: 'assets/mobilenet_v1_1.0_224.txt',
      numThreads: 2,
      isAsset: false,
      useGpuDelegate: true,
    );
    expect(
      log,
      <Matcher>[
        isMethodCall(
          'loadModel',
          arguments: <String, dynamic>{
            'model': 'assets/mobilenet_v1_1.0_224.tflite',
            'labels': 'assets/mobilenet_v1_1.0_224.txt',
            'numThreads': 2,
            'isAsset': false,
            'useGpuDelegate': true,
          },
        ),
      ],
    );
  });

  test('runModelOnImage', () async {
    await Tflite.runModelOnImage(
      path: '/image/path',
      imageMean: 127.5,
      imageStd: 0.5,
      numResults: 6,
      threshold: 0.1,
      asynch: false,
    );
    expect(
      log,
      <Matcher>[
        isMethodCall(
          'runModelOnImage',
          arguments: <String, dynamic>{
            'path': '/image/path',
            'imageMean': 127.5,
            'imageStd': 0.5,
            'numResults': 6,
            'threshold': 0.1,
            'asynch': false,
          },
        ),
      ],
    );
  });

  test('runModelOnBinary', () async {
    await Tflite.runModelOnBinary(
      binary: Uint8List.fromList([
        0,
        1,
        2,
      ]),
      numResults: 15,
      threshold: 0.8,
      asynch: false,
    );
    expect(
      log,
      <Matcher>[
        isMethodCall(
          'runModelOnBinary',
          arguments: <String, dynamic>{
            'binary': Uint8List.fromList([
              0,
              1,
              2,
            ]),
            'numResults': 15,
            'threshold': 0.8,
            'asynch': false,
          },
        ),
      ],
    );
  });

  test('runModelOnFrame', () async {
    await Tflite.runModelOnFrame(
      bytesList: [
        Uint8List.fromList([
          0,
          1,
          2,
        ]),
        Uint8List.fromList([
          0,
          1,
          2,
        ]),
      ],
      imageHeight: 100,
      imageWidth: 200,
      imageMean: 127.5,
      imageStd: 0.5,
      rotation: 30,
      numResults: 10,
      threshold: 0.2,
      asynch: false,
    );
    expect(
      log,
      <Matcher>[
        isMethodCall(
          'runModelOnFrame',
          arguments: <String, dynamic>{
            'bytesList': [
              Uint8List.fromList([
                0,
                1,
                2,
              ]),
              Uint8List.fromList([
                0,
                1,
                2,
              ]),
            ],
            'imageHeight': 100,
            'imageWidth': 200,
            'imageMean': 127.5,
            'imageStd': 0.5,
            'rotation': 30,
            'numResults': 10,
            'threshold': 0.2,
            'asynch': false,
          },
        ),
      ],
    );
  });

  test('detectObjectOnImage', () async {
    await Tflite.detectObjectOnImage(
      path: '/image/path',
      model: 'YOLO',
      imageMean: 127.5,
      imageStd: 0.5,
      threshold: 0.1,
      numResultsPerClass: 5,
      anchors: [
        1,
        2,
        3,
        4,
      ],
      blockSize: 32,
      numBoxesPerBlock: 5,
      asynch: false,
    );
    expect(
      log,
      <Matcher>[
        isMethodCall(
          'detectObjectOnImage',
          arguments: <String, dynamic>{
            'path': '/image/path',
            'model': 'YOLO',
            'imageMean': 127.5,
            'imageStd': 0.5,
            'threshold': 0.1,
            'numResultsPerClass': 5,
            'anchors': [
              1,
              2,
              3,
              4,
            ],
            'blockSize': 32,
            'numBoxesPerBlock': 5,
            'asynch': false,
          },
        ),
      ],
    );
  });

  test('detectObjectOnBinary', () async {
    await Tflite.detectObjectOnBinary(
      binary: Uint8List.fromList([
        0,
        1,
        2,
      ]),
      model: "YOLO",
      threshold: 0.2,
      numResultsPerClass: 10,
      anchors: [
        1,
        2,
        3,
        4,
      ],
      blockSize: 32,
      numBoxesPerBlock: 5,
      asynch: false,
    );
    expect(
      log,
      <Matcher>[
        isMethodCall(
          'detectObjectOnBinary',
          arguments: <String, dynamic>{
            'binary': Uint8List.fromList([
              0,
              1,
              2,
            ]),
            'model': "YOLO",
            'threshold': 0.2,
            'numResultsPerClass': 10,
            'anchors': [
              1,
              2,
              3,
              4,
            ],
            'blockSize': 32,
            'numBoxesPerBlock': 5,
            'asynch': false,
          },
        ),
      ],
    );
  });

  test('detectObjectOnFrame', () async {
    await Tflite.detectObjectOnFrame(
      bytesList: [
        Uint8List.fromList([
          0,
          1,
          2,
        ]),
        Uint8List.fromList([
          0,
          1,
          2,
        ]),
      ],
      model: "YOLO",
      imageHeight: 100,
      imageWidth: 200,
      imageMean: 127.5,
      imageStd: 0.5,
      rotation: 30,
      threshold: 0.2,
      numResultsPerClass: 10,
      anchors: [
        1,
        2,
        3,
        4,
      ],
      blockSize: 32,
      numBoxesPerBlock: 5,
      asynch: false,
    );
    expect(
      log,
      <Matcher>[
        isMethodCall(
          'detectObjectOnFrame',
          arguments: <String, dynamic>{
            'bytesList': [
              Uint8List.fromList([
                0,
                1,
                2,
              ]),
              Uint8List.fromList([
                0,
                1,
                2,
              ]),
            ],
            'model': "YOLO",
            'imageHeight': 100,
            'imageWidth': 200,
            'imageMean': 127.5,
            'imageStd': 0.5,
            'rotation': 30,
            'threshold': 0.2,
            'numResultsPerClass': 10,
            'anchors': [
              1,
              2,
              3,
              4,
            ],
            'blockSize': 32,
            'numBoxesPerBlock': 5,
            'asynch': false,
          },
        ),
      ],
    );
  });

  test('runPix2PixOnImage', () async {
    await Tflite.runPix2PixOnImage(
      path: '/image/path',
      imageMean: 127.5,
      imageStd: 0.5,
      outputType: 'png',
      asynch: false,
    );
    expect(
      log,
      <Matcher>[
        isMethodCall(
          'runPix2PixOnImage',
          arguments: <String, dynamic>{
            'path': '/image/path',
            'imageMean': 127.5,
            'imageStd': 0.5,
            'outputType': 'png',
            'asynch': false,
          },
        ),
      ],
    );
  });

  test('runPix2PixOnBinary', () async {
    await Tflite.runPix2PixOnBinary(
      binary: Uint8List.fromList([
        0,
        1,
        2,
      ]),
      outputType: 'png',
      asynch: false,
    );
    expect(
      log,
      <Matcher>[
        isMethodCall(
          'runPix2PixOnBinary',
          arguments: <String, dynamic>{
            'binary': Uint8List.fromList([
              0,
              1,
              2,
            ]),
            'outputType': 'png',
            'asynch': false,
          },
        ),
      ],
    );
  });

  test('runPix2PixOnFrame', () async {
    await Tflite.runPix2PixOnFrame(
      bytesList: [
        Uint8List.fromList([
          0,
          1,
          2,
        ]),
        Uint8List.fromList([
          0,
          1,
          2,
        ]),
      ],
      imageHeight: 100,
      imageWidth: 200,
      imageMean: 127.5,
      imageStd: 0.5,
      rotation: 30,
      outputType: 'png',
      asynch: false,
    );
    expect(
      log,
      <Matcher>[
        isMethodCall(
          'runPix2PixOnFrame',
          arguments: <String, dynamic>{
            'bytesList': [
              Uint8List.fromList([
                0,
                1,
                2,
              ]),
              Uint8List.fromList([
                0,
                1,
                2,
              ]),
            ],
            'imageHeight': 100,
            'imageWidth': 200,
            'imageMean': 127.5,
            'imageStd': 0.5,
            'rotation': 30,
            'outputType': 'png',
            'asynch': false,
          },
        ),
      ],
    );
  });

  test('runSegmentationOnImage', () async {
    await Tflite.runSegmentationOnImage(
      path: '/image/path',
      imageMean: 127.5,
      imageStd: 0.5,
      labelColors: [
        1,
        2,
        3,
      ],
      outputType: 'png',
      asynch: false,
    );
    expect(
      log,
      <Matcher>[
        isMethodCall(
          'runSegmentationOnImage',
          arguments: <String, dynamic>{
            'path': '/image/path',
            'imageMean': 127.5,
            'imageStd': 0.5,
            'labelColors': [
              1,
              2,
              3,
            ],
            'outputType': 'png',
            'asynch': false,
          },
        ),
      ],
    );
  });

  test('runSegmentationOnBinary', () async {
    await Tflite.runSegmentationOnBinary(
      binary: Uint8List.fromList([
        0,
        1,
        2,
      ]),
      labelColors: [
        1,
        2,
        3,
      ],
      outputType: 'png',
      asynch: false,
    );
    expect(
      log,
      <Matcher>[
        isMethodCall(
          'runSegmentationOnBinary',
          arguments: <String, dynamic>{
            'binary': Uint8List.fromList([
              0,
              1,
              2,
            ]),
            'labelColors': [
              1,
              2,
              3,
            ],
            'outputType': 'png',
            'asynch': false,
          },
        ),
      ],
    );
  });

  test('runSegmentationOnFrame', () async {
    await Tflite.runSegmentationOnFrame(
      bytesList: [
        Uint8List.fromList([
          0,
          1,
          2,
        ]),
        Uint8List.fromList([
          0,
          1,
          2,
        ]),
      ],
      imageHeight: 100,
      imageWidth: 200,
      imageMean: 127.5,
      imageStd: 0.5,
      rotation: 30,
      labelColors: [
        1,
        2,
        3,
      ],
      outputType: 'png',
      asynch: false,
    );
    expect(
      log,
      <Matcher>[
        isMethodCall(
          'runSegmentationOnFrame',
          arguments: <String, dynamic>{
            'bytesList': [
              Uint8List.fromList([
                0,
                1,
                2,
              ]),
              Uint8List.fromList([
                0,
                1,
                2,
              ]),
            ],
            'imageHeight': 100,
            'imageWidth': 200,
            'imageMean': 127.5,
            'imageStd': 0.5,
            'rotation': 30,
            'labelColors': [
              1,
              2,
              3,
            ],
            'outputType': 'png',
            'asynch': false,
          },
        ),
      ],
    );
  });
}
