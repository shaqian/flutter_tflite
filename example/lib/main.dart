import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

import 'package:tflite/tflite.dart';
import 'package:image_picker/image_picker.dart';

void main() => runApp(new MyApp());

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => new _MyAppState();
}

class _MyAppState extends State<MyApp> {
  File _image;
  List _recognitions;

  Future getImage() async {
    var image = await ImagePicker.pickImage(source: ImageSource.camera);
    recognizeImage(image);
    // recognizeImageBinary(image);
    setState(() {
      _image = image;
    });
  }

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future loadModel() async {
    try {
      String res = await Tflite.loadModel(
        model: "assets/mobilenet_v1_1.0_224.tflite",
        labels: "assets/labels.txt",
      );
      print(res);
    } on PlatformException {
      print('Failed to load model.');
    }
  }

  Uint8List imageToByteList(
      img.Image image, int inputSize, double mean, double std) {
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

  Future recognizeImage(File image) async {
    var recognitions = await Tflite.runModelOnImage(
      path: image.path,
      numResults: 6,
      threshold: 0.05,
      imageMean: 127.5,
      imageStd: 127.5,
    );
    print(recognitions);
    setState(() {
      _recognitions = recognitions;
    });
  }

  Future recognizeImageBinary(File image) async {
    var imageBytes = (await rootBundle.load(image.path)).buffer;
    img.Image oriImage = img.decodeJpg(imageBytes.asUint8List());
    img.Image resizedImage = img.copyResize(oriImage, 224, 224);
    var recognitions = await Tflite.runModelOnBinary(
      binary: imageToByteList(resizedImage, 224, 127.5, 127.5),
      numResults: 6,
      threshold: 0.05,
    );
    print(recognitions);
    setState(() {
      _recognitions = recognitions;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('tflite example app'),
        ),
        body: Stack(
          children: <Widget>[
            Center(
              child: _image == null
                  ? Text('No image selected.')
                  : Image.file(_image),
            ),
            Center(
              child: Column(
                children: _recognitions != null
                    ? _recognitions.map((res) {
                        return Text(
                          "${res["index"]} - ${res["label"]}: ${res["confidence"].toString()}",
                          style: TextStyle(
                            color: Colors.black,
                            fontSize: 20.0,
                            background: Paint()..color = Colors.white,
                          ),
                        );
                      }).toList()
                    : [],
              ),
            ),
          ],
        ),
        floatingActionButton: FloatingActionButton(
          onPressed: getImage,
          tooltip: 'Pick Image',
          child: Icon(Icons.add_a_photo),
        ),
      ),
    );
  }
}
