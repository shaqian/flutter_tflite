package sq.flutter.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.BitmapFactory;

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
    } else if (call.method.equals("detectObjectOnImage")) {
      try {
        List<Map<String, Object>> res = detectObjectOnImage((HashMap) call.arguments);
        result.success(res);
      }
      catch (Exception e) {
        result.error("Failed to run model" , e.getMessage(), e);
      }
    } else if (call.method.equals("close")) {
      close();
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

  ByteBuffer feedInputTensorImage(String path, float mean, float std) throws IOException {
    Tensor tensor = tfLite.getInputTensor(0);
    inputSize = tensor.shape()[1];
    int inputChannels = tensor.shape()[3];

    InputStream inputStream = new FileInputStream(path.replace("file://",""));
    Bitmap bitmapRaw = BitmapFactory.decodeStream(inputStream);

    Matrix matrix = getTransformationMatrix(bitmapRaw.getWidth(), bitmapRaw.getHeight(),
        inputSize, inputSize, false);

    int[] intValues = new int[inputSize * inputSize];
    int bytePerChannel = tensor.dataType() == DataType.UINT8 ? 1 : BYTES_PER_CHANNEL;
    ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * inputChannels * bytePerChannel);
    imgData.order(ByteOrder.nativeOrder());

    Bitmap bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);
    final Canvas canvas = new Canvas(bitmap);
    canvas.drawBitmap(bitmapRaw, matrix, null);
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    int pixel = 0;
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[pixel++];
        if (tensor.dataType() == DataType.FLOAT32) {
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - mean) / std);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - mean) / std);
          imgData.putFloat(((pixelValue & 0xFF) - mean) / std);
        } else {
          imgData.put((byte)((pixelValue >> 16) & 0xFF));
          imgData.put((byte)((pixelValue >> 8) & 0xFF));
          imgData.put((byte)(pixelValue & 0xFF));
        }
      }
    }

    return imgData;
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

    tfLite.run(feedInputTensorImage(path, IMAGE_MEAN, IMAGE_STD), labelProb);

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
      int NUM_DETECTIONS = 10;
      float[][][] outputLocations = new float[1][NUM_DETECTIONS][4];
      float[][] outputClasses = new float[1][NUM_DETECTIONS];
      float[][] outputScores = new float[1][NUM_DETECTIONS];
      float[] numDetections = new float[1];

      Object[] inputArray = {imgData};
      Map<Integer, Object> outputMap = new HashMap<>();
      outputMap.put(0, outputLocations);
      outputMap.put(1, outputClasses);
      outputMap.put(2, outputScores);
      outputMap.put(3, numDetections);

      tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

      return parseSSDMobileNet(NUM_DETECTIONS, NUM_RESULTS_PER_CLASS, outputLocations, outputClasses, outputScores);
    } else {
      int gridSize = inputSize / BLOCK_SIZE;
      int numClasses = labels.size();
      final float[][][][] output = new float[1][gridSize][gridSize][(numClasses + 5) * NUM_BOXES_PER_BLOCK];
      tfLite.run(imgData, output);

      return parseYOLO(output, inputSize, BLOCK_SIZE, NUM_BOXES_PER_BLOCK, numClasses, ANCHORS, THRESHOLD, NUM_RESULTS_PER_CLASS);
    }
  }

  private List<Map<String, Object>> parseSSDMobileNet(int numDetections, int numResultsPerClass, float[][][] outputLocations,
                                                      float[][] outputClasses, float[][] outputScores) {
    Map<String, Integer> counters = new HashMap<>();
    final List<Map<String, Object>> results = new ArrayList<>(numDetections);

    for (int i = 0; i < numDetections; ++i) {
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

  private List<Map<String, Object>> parseYOLO(float[][][][] output,
                                              int inputSize,
                                              int blockSize,
                                              int numBoxesPerBlock,
                                              int numClasses,
                                              List<Double> anchors,
                                              float threshold,
                                              int numResultsPerClass) {
    PriorityQueue<Map<String, Object>> pq =
        new PriorityQueue<>(
            1,
            new Comparator<Map<String, Object>>() {
              @Override
              public int compare(Map<String, Object> lhs, Map<String, Object> rhs) {
                return Float.compare((float)rhs.get("confidenceInClass"), (float)lhs.get("confidenceInClass"));
              }
            });

    int gridSize = inputSize / blockSize;

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
