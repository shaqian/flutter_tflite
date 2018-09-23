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

import org.tensorflow.lite.Interpreter;

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
  private Vector<String> labels = new Vector<>();
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
        result.error("Failed to load model" , null, null);
      }
    } if (call.method.equals("runModelOnImage")) {
      try {
        List<Map<String, Object>> res = runModelOnImage((HashMap) call.arguments);
        result.success(res);
      }
      catch (Exception e) {
        result.error("Failed to run model" , null, null);
      }
    } if (call.method.equals("close")) {
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
    tfLite = new Interpreter(buffer);

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
      while ((line = br.readLine()) != null) {
        labels.add(line);
      }
      labelProb = new float[1][labels.size()];
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Failed to read label file" , e);
    }
  }

  private ByteBuffer loadImage(String path, int width, int height, int channels, float mean, float std)
      throws IOException {
    InputStream inputStream = new FileInputStream(path.replace("file://",""));
    Bitmap bitmapRaw = BitmapFactory.decodeStream(inputStream);

    Matrix matrix = getTransformationMatrix(
        bitmapRaw.getWidth(), bitmapRaw.getHeight(),
        width, height, false);

    int[] intValues = new int[width * height];
    ByteBuffer imgData = ByteBuffer.allocateDirect(1 * width * height * channels * BYTES_PER_CHANNEL);
    imgData.order(ByteOrder.nativeOrder());

    Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
    final Canvas canvas = new Canvas(bitmap);
    canvas.drawBitmap(bitmapRaw, matrix, null);
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    int pixel = 0;
    for (int i = 0; i < width; ++i) {
      for (int j = 0; j < height; ++j) {
        int pixelValue = intValues[pixel++];
        imgData.putFloat((((pixelValue >> 16) & 0xFF) - mean) / std);
        imgData.putFloat((((pixelValue >> 8) & 0xFF) - mean) / std);
        imgData.putFloat(((pixelValue & 0xFF) - mean) / std);
      }
    }

    return imgData;
  }

  private List<Map<String, Object>> runModelOnImage(HashMap args) throws IOException {
    String path = args.get("path").toString();
    int NUM_THREADS = (int)args.get("numThreads");
    int WANTED_WIDTH = (int)args.get("inputSize");
    int WANTED_HEIGHT = (int)args.get("inputSize");
    int WANTED_CHANNELS = (int)args.get("numChannels");
    double mean = (double)(args.get("imageMean"));
    float IMAGE_MEAN = (float)mean;
    double std = (double)(args.get("imageStd"));
    float IMAGE_STD = (float)std;
    int NUM_RESULTS = (int)args.get("numResults");
    double threshold = (double)args.get("threshold");
    float THRESHOLD = (float)threshold;

    ByteBuffer imgData = loadImage(path, WANTED_WIDTH, WANTED_HEIGHT, WANTED_CHANNELS, IMAGE_MEAN, IMAGE_STD);

    tfLite.setNumThreads(NUM_THREADS);
    tfLite.run(imgData, labelProb);

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
      if (confidence > THRESHOLD) {
        Map<String, Object> res = new HashMap<>();
        res.put("index", i);
        res.put("label", labels.size() > i ? labels.get(i) : "unknown");
        res.put("confidence", confidence);
        pq.add(res);
      }
    }

    final ArrayList<Map<String, Object>> recognitions = new ArrayList<>();
    int recognitionsSize = Math.min(pq.size(), NUM_RESULTS);
    for (int i = 0; i < recognitionsSize; ++i) {
      recognitions.add(pq.poll());
    }

    return recognitions;
  }


  private void close() {
    tfLite.close();
    labels = null;
    labelProb = null;
  }


  public static Matrix getTransformationMatrix(final int srcWidth,
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
}
