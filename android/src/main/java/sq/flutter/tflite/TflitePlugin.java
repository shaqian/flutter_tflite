package sq.flutter.tflite;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.BitmapFactory;
import android.os.SystemClock;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.util.Log;

import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;
import io.flutter.plugin.common.PluginRegistry.Registrar;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;


public class TflitePlugin implements MethodCallHandler {
  private final Registrar mRegistrar;
  private Interpreter tfLite;
  private int inputSize = 0;
  private Vector<String> labels;
  float[][] labelProb;
  private static final int BYTES_PER_CHANNEL = 4;

  public static void registerWith(Registrar registrar) {
    final MethodChannel channel = new MethodChannel(registrar.messenger(), "tflite");
    channel.setMethodCallHandler(new TflitePlugin(registrar));
  }

  private TflitePlugin(Registrar registrar) {
    this.mRegistrar = registrar;
  }

  @Override
  public void onMethodCall(MethodCall call, Result result) {
    if (call.method.equals("loadModel")) {
      try {
        String res = loadModel((HashMap) call.arguments);
        result.success(res);
      }
      catch (Exception e) {
        result.error("Failed to load model" , e.getMessage(), e);
      }
    } else if (call.method.equals("runModelOnImage")) {
      try {
        List<Map<String, Object>> res = runModelOnImage((HashMap) call.arguments);
        result.success(res);
      }
      catch (Exception e) {
        result.error("Failed to run model" , e.getMessage(), e);
      }
    } else if (call.method.equals("runModelOnBinary")) {
      try {
        List<Map<String, Object>> res = runModelOnBinary((HashMap) call.arguments);
        result.success(res);
      }
      catch (Exception e) {
        result.error("Failed to run model" , e.getMessage(), e);
      }
    } else if (call.method.equals("runModelOnFrame")) {
      try {
        List<Map<String, Object>> res = runModelOnFrame((HashMap) call.arguments);
        result.success(res);
      }
      catch (Exception e) {
        result.error("Failed to run model" , e.getMessage(), e);
      }
    } else if (call.method.equals("detectObjectOnImage")) {
      try {
        List<Map<String, Object>> res = detectObjectOnImage((HashMap) call.arguments);
        result.success(res);
      }
      catch (Exception e) {
        result.error("Failed to run model" , e.getMessage(), e);
      }
    } else if (call.method.equals("detectObjectOnBinary")) {
      try {
        List<Map<String, Object>> res = detectObjectOnBinary((HashMap) call.arguments);
        result.success(res);
      }
      catch (Exception e) {
        result.error("Failed to run model" , e.getMessage(), e);
      }
    } else if (call.method.equals("detectObjectOnFrame")) {
      try {
        List<Map<String, Object>> res = detectObjectOnFrame((HashMap) call.arguments);
        result.success(res);
      }
      catch (Exception e) {
        result.error("Failed to run model" , e.getMessage(), e);
      }
    } else if (call.method.equals("close")) {
      close();
    } else if (call.method.equals("runPix2PixOnImage")) {
      try {
        List<Map<String, Object>> res = runPix2PixOnImage((HashMap) call.arguments);
        result.success(res);
      }
      catch (Exception e) {
        result.error("Failed to run model" , e.getMessage(), e);
      }
    } else if (call.method.equals("runPix2PixOnBinary")) {
      try {
        List<Map<String, Object>> res = runPix2PixOnBinary((HashMap) call.arguments);
        result.success(res);
      }
      catch (Exception e) {
        result.error("Failed to run model" , e.getMessage(), e);
      }
    } else if (call.method.equals("runPix2PixOnFrame")) {
      try {
        List<Map<String, Object>> res = runPix2PixOnFrame((HashMap) call.arguments);
        result.success(res);
      }
      catch (Exception e) {
        result.error("Failed to run model" , e.getMessage(), e);
      }
    } else if (call.method.equals("runSegmentationOnImage")) {
      try {
        byte[] res = runSegmentationOnImage((HashMap) call.arguments);
        result.success(res);
      }
      catch (Exception e) {
        result.error("Failed to run model" , e.getMessage(), e);
      }
    } else {
      result.error("Invalid method", call.method.toString(), "");
    }
  }

  private String loadModel(HashMap args) throws IOException {
    String model = args.get("model").toString();
    AssetManager assetManager = mRegistrar.context().getAssets();
    String key = mRegistrar.lookupKeyForAsset(model);
    AssetFileDescriptor fileDescriptor = assetManager.openFd(key);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

    int numThreads = (int)args.get("numThreads");
    final Interpreter.Options tfliteOptions = new Interpreter.Options();
    tfliteOptions.setNumThreads(numThreads);
    tfLite = new Interpreter(buffer, tfliteOptions);

    String labels = args.get("labels").toString();
    key = mRegistrar.lookupKeyForAsset(labels);
    loadLabels(assetManager, key);

    return "success";
  }

  private void loadLabels(AssetManager assetManager, String path) {
    BufferedReader br;
    try {
      br = new BufferedReader(new InputStreamReader(assetManager.open(path)));
      String line;
      labels = new Vector<>();
      while ((line = br.readLine()) != null) {
        labels.add(line);
      }
      labelProb = new float[1][labels.size()];
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Failed to read label file" , e);
    }
  }

  private List<Map<String, Object>> GetTopN(int numResults, float threshold) {
    PriorityQueue<Map<String, Object>> pq =
        new PriorityQueue<>(
            1,
            new Comparator<Map<String, Object>>() {
              @Override
              public int compare(Map<String, Object> lhs, Map<String, Object> rhs) {
                return Float.compare((float)rhs.get("confidence"), (float)lhs.get("confidence"));
              }
            });

    for (int i = 0; i < labels.size(); ++i) {
      float confidence = labelProb[0][i];
      if (confidence > threshold) {
        Map<String, Object> res = new HashMap<>();
        res.put("index", i);
        res.put("label", labels.size() > i ? labels.get(i) : "unknown");
        res.put("confidence", confidence);
        pq.add(res);
      }
    }

    final ArrayList<Map<String, Object>> recognitions = new ArrayList<>();
    int recognitionsSize = Math.min(pq.size(), numResults);
    for (int i = 0; i < recognitionsSize; ++i) {
      recognitions.add(pq.poll());
    }

    return recognitions;
  }

  Bitmap feedOutput(ByteBuffer imgData, float mean, float std) {
    Tensor tensor = tfLite.getOutputTensor(0);
    int outputSize = tensor.shape()[1];
    Bitmap bitmapRaw = Bitmap.createBitmap(outputSize, outputSize, Bitmap.Config.ARGB_8888);

    if (tensor.dataType() == DataType.FLOAT32) {
      for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
          int pixelValue = 0xFF << 24;
          pixelValue |= ((Math.round(imgData.getFloat() * std + mean) & 0xFF) << 16);
          pixelValue |= ((Math.round(imgData.getFloat() * std + mean) & 0xFF) << 8);
          pixelValue |= ((Math.round(imgData.getFloat() * std + mean) & 0xFF));
          bitmapRaw.setPixel(j, i, pixelValue);
        }
      }
    } else {
      for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
          int pixelValue = 0xFF << 24;
          pixelValue |= ((imgData.get() & 0xFF) << 16);
          pixelValue |= ((imgData.get() & 0xFF) << 8);
          pixelValue |= ((imgData.get() & 0xFF));
          bitmapRaw.setPixel(j, i, pixelValue);
        }
      }
    }
    return bitmapRaw;
  }

  ByteBuffer feedInputTensor(Bitmap bitmapRaw, float mean, float std) throws IOException {
    Tensor tensor = tfLite.getInputTensor(0);
    inputSize = tensor.shape()[1];
    int inputChannels = tensor.shape()[3];

    int bytePerChannel = tensor.dataType() == DataType.UINT8 ? 1 : BYTES_PER_CHANNEL;
    ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * inputChannels * bytePerChannel);
    imgData.order(ByteOrder.nativeOrder());

    Bitmap bitmap = bitmapRaw;
    if (bitmapRaw.getWidth() != inputSize || bitmapRaw.getHeight() != inputSize) {
      Matrix matrix = getTransformationMatrix(bitmapRaw.getWidth(), bitmapRaw.getHeight(),
                                              inputSize, inputSize, false);
      bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);
      final Canvas canvas = new Canvas(bitmap);
      canvas.drawBitmap(bitmapRaw, matrix, null);
    }

    if (tensor.dataType() == DataType.FLOAT32) {
      for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
          int pixelValue = bitmap.getPixel(j, i);
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - mean) / std);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - mean) / std);
          imgData.putFloat(((pixelValue & 0xFF) - mean) / std);
        }
      }
    } else {
      for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
          int pixelValue = bitmap.getPixel(j, i);
          imgData.put((byte)((pixelValue >> 16) & 0xFF));
          imgData.put((byte)((pixelValue >> 8) & 0xFF));
          imgData.put((byte)(pixelValue & 0xFF));
        }
      }
    }

    return imgData;
  }

  ByteBuffer feedInputTensorImage(String path, float mean, float std) throws IOException {
    InputStream inputStream = new FileInputStream(path.replace("file://",""));
    Bitmap bitmapRaw = BitmapFactory.decodeStream(inputStream);

    return feedInputTensor(bitmapRaw, mean, std);
  }

  ByteBuffer feedInputTensorFrame(List<byte[]> bytesList, int imageHeight, int imageWidth, float mean, float std, int rotation) throws IOException {
    ByteBuffer Y = ByteBuffer.wrap(bytesList.get(0));
    ByteBuffer U = ByteBuffer.wrap(bytesList.get(1));
    ByteBuffer V = ByteBuffer.wrap(bytesList.get(2));

    int Yb = Y.remaining();
    int Ub = U.remaining();
    int Vb = V.remaining();

    byte[] data = new byte[Yb + Ub + Vb];

    Y.get(data, 0, Yb);
    V.get(data, Yb, Vb);
    U.get(data, Yb + Vb, Ub);

    Bitmap bitmapRaw = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);
    Allocation bmData = renderScriptNV21ToRGBA888(
        mRegistrar.context(),
        imageWidth,
        imageHeight,
        data);
    bmData.copyTo(bitmapRaw);

    Matrix matrix = new Matrix();
    matrix.postRotate(rotation);
    bitmapRaw = Bitmap.createBitmap(bitmapRaw, 0, 0, bitmapRaw.getWidth(), bitmapRaw.getHeight(), matrix, true);

    return feedInputTensor(bitmapRaw, mean, std);
  }

  public Allocation renderScriptNV21ToRGBA888(Context context, int width, int height, byte[] nv21) {
    // https://stackoverflow.com/a/36409748
    RenderScript rs = RenderScript.create(context);
    ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));

    Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(nv21.length);
    Allocation in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);

    Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
    Allocation out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);

    in.copyFrom(nv21);

    yuvToRgbIntrinsic.setInput(in);
    yuvToRgbIntrinsic.forEach(out);
    return out;
  }

  private List<Map<String, Object>> runModelOnImage(HashMap args) throws IOException {
    String path = args.get("path").toString();
    double mean = (double)(args.get("imageMean"));
    float IMAGE_MEAN = (float)mean;
    double std = (double)(args.get("imageStd"));
    float IMAGE_STD = (float)std;
    int NUM_RESULTS = (int)args.get("numResults");
    double threshold = (double)args.get("threshold");
    float THRESHOLD = (float)threshold;

    long startTime = SystemClock.uptimeMillis();
    tfLite.run(feedInputTensorImage(path, IMAGE_MEAN, IMAGE_STD), labelProb);
    Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));

    return GetTopN(NUM_RESULTS, THRESHOLD);
  }

  private List<Map<String, Object>> runModelOnBinary(HashMap args) throws IOException {
    byte[] binary = (byte[])args.get("binary");
    int NUM_RESULTS = (int)args.get("numResults");
    double threshold = (double)args.get("threshold");
    float THRESHOLD = (float)threshold;

    ByteBuffer imgData = ByteBuffer.wrap(binary);
    tfLite.run(imgData, labelProb);

    return GetTopN(NUM_RESULTS, THRESHOLD);
  }

  private List<Map<String, Object>> runModelOnFrame(HashMap args) throws IOException {
    List<byte[]> bytesList= (ArrayList)args.get("bytesList");
    double mean = (double)(args.get("imageMean"));
    float IMAGE_MEAN = (float)mean;
    double std = (double)(args.get("imageStd"));
    float IMAGE_STD = (float)std;
    int imageHeight = (int)(args.get("imageHeight"));
    int imageWidth = (int)(args.get("imageWidth"));
    int rotation = (int)(args.get("rotation"));
    int NUM_RESULTS = (int)args.get("numResults");
    double threshold = (double)args.get("threshold");
    float THRESHOLD = (float)threshold;

    long startTime = SystemClock.uptimeMillis();

    ByteBuffer imgData = feedInputTensorFrame(bytesList, imageHeight, imageWidth, IMAGE_MEAN, IMAGE_STD, rotation);

    tfLite.run(imgData, labelProb);

    Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));

    return GetTopN(NUM_RESULTS, THRESHOLD);
  }

  private List<Map<String, Object>> detectObjectOnImage(HashMap args) throws IOException {
    String path = args.get("path").toString();
    String model = args.get("model").toString();
    double mean = (double)(args.get("imageMean"));
    float IMAGE_MEAN = (float)mean;
    double std = (double)(args.get("imageStd"));
    float IMAGE_STD = (float)std;
    double threshold = (double)args.get("threshold");
    float THRESHOLD = (float)threshold;
    List<Double> ANCHORS = (ArrayList)args.get("anchors");
    int BLOCK_SIZE = (int)args.get("blockSize");
    int NUM_BOXES_PER_BLOCK = (int)args.get("numBoxesPerBlock");
    int NUM_RESULTS_PER_CLASS = (int)args.get("numResultsPerClass");

    ByteBuffer imgData = feedInputTensorImage(path, IMAGE_MEAN, IMAGE_STD);

    if (model.equals("SSDMobileNet")) {
      return parseSSDMobileNet(imgData, NUM_RESULTS_PER_CLASS, THRESHOLD);
    } else {
      return parseYOLO(imgData, BLOCK_SIZE, NUM_BOXES_PER_BLOCK, ANCHORS, THRESHOLD, NUM_RESULTS_PER_CLASS);
    }
  }

  private List<Map<String, Object>> detectObjectOnBinary(HashMap args) throws IOException {
    byte[] binary = (byte[])args.get("binary");
    String model = args.get("model").toString();
    double threshold = (double)args.get("threshold");
    float THRESHOLD = (float)threshold;
    List<Double> ANCHORS = (ArrayList)args.get("anchors");
    int BLOCK_SIZE = (int)args.get("blockSize");
    int NUM_BOXES_PER_BLOCK = (int)args.get("numBoxesPerBlock");
    int NUM_RESULTS_PER_CLASS = (int)args.get("numResultsPerClass");

    ByteBuffer imgData = ByteBuffer.wrap(binary);

    if (model.equals("SSDMobileNet")) {
      return parseSSDMobileNet(imgData, NUM_RESULTS_PER_CLASS, THRESHOLD);
    } else {
      return parseYOLO(imgData, BLOCK_SIZE, NUM_BOXES_PER_BLOCK, ANCHORS, THRESHOLD, NUM_RESULTS_PER_CLASS);
    }
  }

  private List<Map<String, Object>> detectObjectOnFrame(HashMap args) throws IOException {
    List<byte[]> bytesList= (ArrayList)args.get("bytesList");
    String model = args.get("model").toString();
    double mean = (double)(args.get("imageMean"));
    float IMAGE_MEAN = (float)mean;
    double std = (double)(args.get("imageStd"));
    float IMAGE_STD = (float)std;
    int imageHeight = (int)(args.get("imageHeight"));
    int imageWidth = (int)(args.get("imageWidth"));
    int rotation = (int)(args.get("rotation"));
    double threshold = (double)args.get("threshold");
    float THRESHOLD = (float)threshold;
    int NUM_RESULTS_PER_CLASS = (int)args.get("numResultsPerClass");

    List<Double> ANCHORS = (ArrayList)args.get("anchors");
    int BLOCK_SIZE = (int)args.get("blockSize");
    int NUM_BOXES_PER_BLOCK = (int)args.get("numBoxesPerBlock");

    ByteBuffer imgData = feedInputTensorFrame(bytesList, imageHeight, imageWidth, IMAGE_MEAN, IMAGE_STD, rotation);

    if (model.equals("SSDMobileNet")) {
      return parseSSDMobileNet(imgData, NUM_RESULTS_PER_CLASS, THRESHOLD);
    } else {
      return parseYOLO(imgData, BLOCK_SIZE, NUM_BOXES_PER_BLOCK, ANCHORS, THRESHOLD, NUM_RESULTS_PER_CLASS);
    }
  }

  private List<Map<String, Object>> runPix2PixOnImage(HashMap args) throws IOException {
    String path = args.get("path").toString();
    double mean = (double)(args.get("imageMean"));
    float IMAGE_MEAN = (float)mean;
    double std = (double)(args.get("imageStd"));
    float IMAGE_STD = (float)std;

    long startTime = SystemClock.uptimeMillis();
    ByteBuffer input = feedInputTensorImage(path, IMAGE_MEAN, IMAGE_STD);
    ByteBuffer output = ByteBuffer.allocateDirect(input.limit());
    output.order(ByteOrder.nativeOrder());
    if (input.limit() == 0) throw new RuntimeException("Unexpected input position, bad file?");
    if (output.position() != 0) throw new RuntimeException("Unexpected output position");
    tfLite.run(input, output);
    if (output.position() != input.limit()) throw new RuntimeException("Mismatching input/output position");

    output.flip();
    Bitmap bitmapRaw = feedOutput(output, IMAGE_MEAN, IMAGE_STD);
    String fileExt = path.substring(path.lastIndexOf('.')+1);
    String outputFilename = path.substring(0, path.lastIndexOf('.')) + "_pix2pix." + fileExt;
    try (FileOutputStream out = new FileOutputStream(outputFilename, false)) {
      bitmapRaw.compress(Bitmap.CompressFormat.PNG, 100, out);
    } catch (IOException e) {
      e.printStackTrace();
      outputFilename = "";
    }

    final ArrayList<Map<String, Object>> result = new ArrayList<>();
    Map<String, Object> res = new HashMap<>();
    res.put("filename", outputFilename);
    result.add(res);
    return result;
  }

  private List<Map<String, Object>> runPix2PixOnBinary(HashMap args) throws IOException {
    byte[] binary = (byte[])args.get("binary");

    long startTime = SystemClock.uptimeMillis();
    ByteBuffer input = ByteBuffer.wrap(binary);
    ByteBuffer output = ByteBuffer.allocateDirect(input.limit());
    output.order(ByteOrder.nativeOrder());

    if (input.limit() == 0) throw new RuntimeException("Unexpected input position, bad file?");
    if (output.position() != 0) throw new RuntimeException("Unexpected output position");
    tfLite.run(input, output);
    Log.v("time", "Generating took " + (SystemClock.uptimeMillis() - startTime));
    if (output.position() != input.limit()) throw new RuntimeException("Mismatching input/output position");

    final ArrayList<Map<String, Object>> result = new ArrayList<>();
    Map<String, Object> res = new HashMap<>();
    res.put("binary", output.array());
    result.add(res);
    return result;
  }

  private List<Map<String, Object>> runPix2PixOnFrame(HashMap args) throws IOException {
    List<byte[]> bytesList= (ArrayList)args.get("bytesList");
    double mean = (double)(args.get("imageMean"));
    float IMAGE_MEAN = (float)mean;
    double std = (double)(args.get("imageStd"));
    float IMAGE_STD = (float)std;
    int imageHeight = (int)(args.get("imageHeight"));
    int imageWidth = (int)(args.get("imageWidth"));
    int rotation = (int)(args.get("rotation"));

    long startTime = SystemClock.uptimeMillis();
    ByteBuffer input = feedInputTensorFrame(bytesList, imageHeight, imageWidth, IMAGE_MEAN, IMAGE_STD, rotation);
    ByteBuffer output = ByteBuffer.allocateDirect(input.limit());
    output.order(ByteOrder.nativeOrder());

    if (input.limit() == 0) throw new RuntimeException("Unexpected input position, bad file?");
    if (output.position() != 0) throw new RuntimeException("Unexpected output position");
    tfLite.run(input, output);
    Log.v("time", "Generating took " + (SystemClock.uptimeMillis() - startTime));
    if (output.position() != input.limit()) throw new RuntimeException("Mismatching input/output position");

    final ArrayList<Map<String, Object>> result = new ArrayList<>();
    Map<String, Object> res = new HashMap<>();
    res.put("binary", output.array());
    result.add(res);
    return result;
  }

  private List<Map<String, Object>> parseSSDMobileNet(ByteBuffer imgData, int numResultsPerClass, float threshold) {
    int num = tfLite.getOutputTensor(0).shape()[1];
    float[][][] outputLocations = new float[1][num][4];
    float[][] outputClasses = new float[1][num];
    float[][] outputScores = new float[1][num];
    float[] numDetections = new float[1];

    Object[] inputArray = {imgData};
    Map<Integer, Object> outputMap = new HashMap<>();
    outputMap.put(0, outputLocations);
    outputMap.put(1, outputClasses);
    outputMap.put(2, outputScores);
    outputMap.put(3, numDetections);

    long startTime = SystemClock.uptimeMillis();

    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

    Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));

    Map<String, Integer> counters = new HashMap<>();
    final List<Map<String, Object>> results = new ArrayList<>();

    for (int i = 0; i < numDetections[0]; ++i) {
      if (outputScores[0][i] < threshold) continue;

      String detectedClass = labels.get((int) outputClasses[0][i] + 1);

      if (counters.get(detectedClass) == null) {
        counters.put(detectedClass, 1);
      } else {
        int count = counters.get(detectedClass);
        if (count >= numResultsPerClass) {
          continue;
        } else {
          counters.put(detectedClass, count + 1);
        }
      }

      Map<String, Object> rect = new HashMap<>();
      float ymin = Math.max(0, outputLocations[0][i][0]);
      float xmin = Math.max(0, outputLocations[0][i][1]);
      float ymax = outputLocations[0][i][2];
      float xmax = outputLocations[0][i][3];
      rect.put("x", xmin);
      rect.put("y", ymin);
      rect.put("w", Math.min(1 - xmin, xmax - xmin));
      rect.put("h", Math.min(1 - ymin, ymax - ymin));

      Map<String, Object> result = new HashMap<>();
      result.put("rect", rect);
      result.put("confidenceInClass", outputScores[0][i]);
      result.put("detectedClass", detectedClass);

      results.add(result);
    }

    return results;
  }

  private List<Map<String, Object>> parseYOLO(ByteBuffer imgData,
                                              int blockSize,
                                              int numBoxesPerBlock,
                                              List<Double> anchors,
                                              float threshold,
                                              int numResultsPerClass) {
    long startTime = SystemClock.uptimeMillis();

    Tensor tensor = tfLite.getInputTensor(0);
    inputSize = tensor.shape()[1];
    int gridSize = inputSize / blockSize;
    int numClasses = labels.size();
    final float[][][][] output = new float[1][gridSize][gridSize][(numClasses + 5) * numBoxesPerBlock];
    tfLite.run(imgData, output);

    Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));

    PriorityQueue<Map<String, Object>> pq =
        new PriorityQueue<>(
            1,
            new Comparator<Map<String, Object>>() {
              @Override
              public int compare(Map<String, Object> lhs, Map<String, Object> rhs) {
                return Float.compare((float)rhs.get("confidenceInClass"), (float)lhs.get("confidenceInClass"));
              }
            });

    for (int y = 0; y < gridSize; ++y) {
      for (int x = 0; x < gridSize; ++x) {
        for (int b = 0; b < numBoxesPerBlock; ++b) {
          final int offset = (numClasses + 5) * b;

          final float confidence = expit(output[0][y][x][offset + 4]);

          final float[] classes = new float[numClasses];
          for (int c = 0; c < numClasses; ++c) {
            classes[c] = output[0][y][x][offset + 5 + c];
          }
          softmax(classes);

          int detectedClass = -1;
          float maxClass = 0;
          for (int c = 0; c < numClasses; ++c) {
            if (classes[c] > maxClass) {
              detectedClass = c;
              maxClass = classes[c];
            }
          }

          final float confidenceInClass = maxClass * confidence;
          if (confidenceInClass > threshold) {
            final float xPos = (x + expit(output[0][y][x][offset + 0])) * blockSize;
            final float yPos = (y + expit(output[0][y][x][offset + 1])) * blockSize;

            final float w = (float) (Math.exp(output[0][y][x][offset + 2]) * anchors.get(2 * b + 0)) * blockSize;
            final float h = (float) (Math.exp(output[0][y][x][offset + 3]) * anchors.get(2 * b + 1)) * blockSize;

            final float xmin = Math.max(0, (xPos - w / 2) / inputSize);
            final float ymin = Math.max(0, (yPos - h / 2) / inputSize);

            Map<String, Object> rect = new HashMap<>();
            rect.put("x", xmin);
            rect.put("y", ymin);
            rect.put("w", Math.min(1 - xmin, w / inputSize));
            rect.put("h", Math.min(1 - ymin, h / inputSize));

            Map<String, Object> result = new HashMap<>();
            result.put("rect", rect);
            result.put("confidenceInClass", confidenceInClass);
            result.put("detectedClass", labels.get(detectedClass));

            pq.add(result);
          }
        }
      }
    }

    Map<String, Integer> counters = new HashMap<>();
    List<Map<String, Object>> results = new ArrayList<>();

    for (int i = 0; i < pq.size(); ++i) {
      Map<String, Object> result = pq.poll();
      String detectedClass = result.get("detectedClass").toString();

      if (counters.get(detectedClass) == null) {
        counters.put(detectedClass, 1);
      } else {
        int count = counters.get(detectedClass);
        if (count >= numResultsPerClass) {
          continue;
        } else {
          counters.put(detectedClass, count + 1);
        }
      }
      results.add(result);
    }
    return results;
  }

  private byte[] runSegmentationOnImage(HashMap args) throws IOException {
    String path = args.get("path").toString();
    double mean = (double)(args.get("imageMean"));
    float IMAGE_MEAN = (float)mean;
    double std = (double)(args.get("imageStd"));
    float IMAGE_STD = (float)std;
    List<Long> labelColors = (ArrayList)args.get("labelColors");

    long startTime = SystemClock.uptimeMillis();
    ByteBuffer input = feedInputTensorImage(path, IMAGE_MEAN, IMAGE_STD);
    ByteBuffer output = ByteBuffer.allocateDirect(tfLite.getOutputTensor(0).numBytes());
    output.order(ByteOrder.nativeOrder());
    tfLite.run(input, output);
    Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));

    if (input.limit() == 0) throw new RuntimeException("Unexpected input position, bad file?");
    if (output.position() != output.limit()) throw new RuntimeException("Unexpected output position");

    output.flip();
    Bitmap outputArgmax = fetchArgmax(output, labelColors);
    return compressPNG(outputArgmax);
  }


  Bitmap fetchArgmax(ByteBuffer output, List<Long> labelColors) {
    Tensor outputTensor = tfLite.getOutputTensor(0);
    int outputBatchSize = outputTensor.shape()[0];
    assert outputBatchSize == 1;
    int outputHeight = outputTensor.shape()[1];
    int outputWidth = outputTensor.shape()[2];
    int outputChannels = outputTensor.shape()[3];

    Bitmap outputArgmax = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888);

    if (outputTensor.dataType() == DataType.FLOAT32) {
      for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
          int maxIndex = 0;
          float maxValue = 0.0f;
          for (int c = 0; c < outputChannels; ++c) {
            float outputValue = output.getFloat();
            if (outputValue > maxValue) {
              maxIndex = c;
              maxValue = outputValue;
            }
          }
          int labelColor = labelColors.get(maxIndex).intValue();
          outputArgmax.setPixel(j, i, labelColor);
        }
      }
    } else {
      for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
          int maxIndex = 0;
          int maxValue = 0;
          for (int c = 0; c < outputChannels; ++c) {
            int outputValue = output.get();
            if (outputValue > maxValue) {
              maxIndex = c;
              maxValue = outputValue;
            }
          }
          int labelColor = labelColors.get(maxIndex).intValue();
          outputArgmax.setPixel(j, i, labelColor);
        }
      }
    }
    return outputArgmax;
  }

  byte[] compressPNG(Bitmap bitmap) {
    // https://stackoverflow.com/questions/4989182/converting-java-bitmap-to-byte-array#4989543
    ByteArrayOutputStream stream = new ByteArrayOutputStream();
    bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
    byte[] byteArray = stream.toByteArray();
    // bitmap.recycle();
    return byteArray;
  }

  private float expit(final float x) {
    return (float) (1. / (1. + Math.exp(-x)));
  }

  private void softmax(final float[] vals) {
    float max = Float.NEGATIVE_INFINITY;
    for (final float val : vals) {
      max = Math.max(max, val);
    }
    float sum = 0.0f;
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = (float) Math.exp(vals[i] - max);
      sum += vals[i];
    }
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = vals[i] / sum;
    }
  }

  private static Matrix getTransformationMatrix(final int srcWidth,
                                                final int srcHeight,
                                                final int dstWidth,
                                                final int dstHeight,
                                                final boolean maintainAspectRatio)
  {
    final Matrix matrix = new Matrix();

    if (srcWidth != dstWidth || srcHeight != dstHeight) {
      final float scaleFactorX = dstWidth / (float) srcWidth;
      final float scaleFactorY = dstHeight / (float) srcHeight;

      if (maintainAspectRatio) {
        final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
        matrix.postScale(scaleFactor, scaleFactor);
      } else {
        matrix.postScale(scaleFactorX, scaleFactorY);
      }
    }

    matrix.invert(new Matrix());
    return matrix;
  }

  private void close() {
    tfLite.close();
    labels = null;
    labelProb = null;
  }
}
