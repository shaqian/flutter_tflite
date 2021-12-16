//#define CONTRIB_PATH
#define TFLITE2

#import "TflitePlugin.h"

#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#ifdef CONTRIB_PATH
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/op_resolver.h"
#elif defined TFLITE2
#import "TensorFlowLiteC.h"
#import "metal_delegate.h"
#else
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/op_resolver.h"
#endif

#include "ios_image_load.h"

#define LOG(x) std::cerr

typedef void (^TfLiteStatusCallback)(TfLiteStatus);
NSString* loadModel(NSObject<FlutterPluginRegistrar>* _registrar, NSDictionary* args);
void runTflite(NSDictionary* args, TfLiteStatusCallback cb);
void runModelOnImage(NSDictionary* args, FlutterResult result);
void runModelOnBinary(NSDictionary* args, FlutterResult result);
void runModelOnFrame(NSDictionary* args, FlutterResult result);
void detectObjectOnImage(NSDictionary* args, FlutterResult result);
void detectObjectOnBinary(NSDictionary* args, FlutterResult result);
void detectObjectOnFrame(NSDictionary* args, FlutterResult result);
void runPix2PixOnImage(NSDictionary* args, FlutterResult result);
void runPix2PixOnBinary(NSDictionary* args, FlutterResult result);
void runPix2PixOnFrame(NSDictionary* args, FlutterResult result);
void runSegmentationOnImage(NSDictionary* args, FlutterResult result);
void runSegmentationOnBinary(NSDictionary* args, FlutterResult result);
void runSegmentationOnFrame(NSDictionary* args, FlutterResult result);
void runPoseNetOnImage(NSDictionary* args, FlutterResult result);
void runPoseNetOnBinary(NSDictionary* args, FlutterResult result);
void runPoseNetOnFrame(NSDictionary* args, FlutterResult result);
void close();

@implementation TflitePlugin {
  NSObject<FlutterPluginRegistrar>* _registrar;
}

+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  FlutterMethodChannel* channel = [FlutterMethodChannel
      methodChannelWithName:@"tflite"
            binaryMessenger:[registrar messenger]];
  TflitePlugin* instance = [[TflitePlugin alloc] initWithRegistrar:registrar];
  [registrar addMethodCallDelegate:instance channel:channel];
}

- (instancetype)initWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  self = [super init];
  if (self) {
    _registrar = registrar;
  }
  return self;
}

- (void)handleMethodCall:(FlutterMethodCall*)call result:(FlutterResult)result {
  if ([@"loadModel" isEqualToString:call.method]) {
    NSString* load_result = loadModel(_registrar, call.arguments);
    result(load_result);
  } else if ([@"runModelOnImage" isEqualToString:call.method]) {
    runModelOnImage(call.arguments, result);
  } else if ([@"runModelOnBinary" isEqualToString:call.method]) {
    runModelOnBinary(call.arguments, result);
  } else if ([@"runModelOnFrame" isEqualToString:call.method]) {
    runModelOnFrame(call.arguments, result);
  } else if ([@"detectObjectOnImage" isEqualToString:call.method]) {
    detectObjectOnImage(call.arguments, result);
  } else if ([@"detectObjectOnBinary" isEqualToString:call.method]) {
    detectObjectOnBinary(call.arguments, result);
  } else if ([@"detectObjectOnFrame" isEqualToString:call.method]) {
    detectObjectOnFrame(call.arguments, result);
  } else if ([@"runPix2PixOnImage" isEqualToString:call.method]) {
    runPix2PixOnImage(call.arguments, result);
  } else if ([@"runPix2PixOnBinary" isEqualToString:call.method]) {
    runPix2PixOnBinary(call.arguments, result);
  } else if ([@"runPix2PixOnFrame" isEqualToString:call.method]) {
    runPix2PixOnFrame(call.arguments, result);
  } else if ([@"runSegmentationOnImage" isEqualToString:call.method]) {
    runSegmentationOnImage(call.arguments, result);
  } else if ([@"runSegmentationOnBinary" isEqualToString:call.method]) {
    runSegmentationOnBinary(call.arguments, result);
  } else if ([@"runSegmentationOnFrame" isEqualToString:call.method]) {
    runSegmentationOnFrame(call.arguments, result);
  } else if ([@"runPoseNetOnImage" isEqualToString:call.method]) {
    runPoseNetOnImage(call.arguments, result);
  } else if ([@"runPoseNetOnBinary" isEqualToString:call.method]) {
    runPoseNetOnBinary(call.arguments, result);
  } else if ([@"runPoseNetOnFrame" isEqualToString:call.method]) {
    runPoseNetOnFrame(call.arguments, result);
  } else if ([@"close" isEqualToString:call.method]) {
    close();
  } else {
    result(FlutterMethodNotImplemented);
  }
}

@end

std::vector<std::string> labels;
#ifdef TFLITE2
TfLiteInterpreter *interpreter = nullptr;
TfLiteModel *model = nullptr;
TfLiteDelegate *delegate = nullptr;
#else
std::unique_ptr<tflite::FlatBufferModel> model;
std::unique_ptr<tflite::Interpreter> interpreter;
#endif
bool interpreter_busy = false;

static void LoadLabels(NSString* labels_path,
                       std::vector<std::string>* label_strings) {
  if (!labels_path) {
    LOG(ERROR) << "Failed to find label file at" << labels_path;
  }
  std::ifstream t;
  t.open([labels_path UTF8String]);
  label_strings->clear();
  for (std::string line; std::getline(t, line); ) {
    label_strings->push_back(line);
  }
  t.close();
}

NSString* loadModel(NSObject<FlutterPluginRegistrar>* _registrar, NSDictionary* args) {
  NSString* graph_path;
  NSString* key;
  NSNumber* isAssetNumber = args[@"isAsset"];
  bool isAsset = [isAssetNumber boolValue];
  if(isAsset){
    key = [_registrar lookupKeyForAsset:args[@"model"]];
    graph_path = [[NSBundle mainBundle] pathForResource:key ofType:nil];
  }else{
    graph_path = args[@"model"];
  }

  const int num_threads = [args[@"numThreads"] intValue];
  
#ifdef TFLITE2
  TfLiteInterpreterOptions *options = nullptr;
  model = TfLiteModelCreateFromFile(graph_path.UTF8String);
  if (!model) {
    return [NSString stringWithFormat:@"%s %@", "Failed to mmap model", graph_path];
  }
  options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetNumThreads(options, num_threads);
  
  bool useGpuDelegate = [args[@"useGpuDelegate"] boolValue];
  if (useGpuDelegate) {
    delegate = TFLGpuDelegateCreate(nullptr);
    TfLiteInterpreterOptionsAddDelegate(options, delegate);
  }
#else
  model = tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]);
  if (!model) {
    return [NSString stringWithFormat:@"%s %@", "Failed to mmap model", graph_path];
  }
  LOG(INFO) << "Loaded model " << graph_path;
  model->error_reporter();
  LOG(INFO) << "resolved reporter";
#endif
  
  if ([args[@"labels"] length] > 0) {
    NSString* labels_path;
    if(isAsset){
      key = [_registrar lookupKeyForAsset:args[@"labels"]];
      labels_path = [[NSBundle mainBundle] pathForResource:key ofType:nil];
    }else{
      labels_path = args[@"labels"];
    }
    LoadLabels(labels_path, &labels);
  }

#ifdef TFLITE2
  interpreter = TfLiteInterpreterCreate(model, options);
  if (!interpreter) {
    return @"Failed to construct interpreter";
  }
  
  if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
     return @"Failed to allocate tensors!";
   }
#else
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    return @"Failed to construct interpreter";
  }
  
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    return @"Failed to allocate tensors!";
  }
  
  if (num_threads != -1) {
    interpreter->SetNumThreads(num_threads);
  }
  #endif
  
  return @"success";
}


void runTflite(NSDictionary* args, TfLiteStatusCallback cb) {
  const bool asynch = [args[@"asynch"] boolValue];
  if (asynch) {
    interpreter_busy = true;
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(void){
#ifdef TFLITE2
      TfLiteStatus status = TfLiteInterpreterInvoke(interpreter);
#else
      TfLiteStatus status = interpreter->Invoke();
#endif
      dispatch_async(dispatch_get_main_queue(), ^(void){
        interpreter_busy = false;
        cb(status);
      });
    });
  } else {
#ifdef TFLITE2
    TfLiteStatus status = TfLiteInterpreterInvoke(interpreter);
#else
    TfLiteStatus status = interpreter->Invoke();
#endif
    cb(status);
  }
}

NSMutableData *feedOutputTensor(int outputChannelsIn, float mean, float std, bool convertToUint8,
                                int *widthOut, int *heightOut) {
#ifdef TFLITE2
  assert(TfLiteInterpreterGetOutputTensorCount(interpreter) == 1);
  const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
#else
  assert(interpreter->outputs().size() == 1);
  int output = interpreter->outputs()[0];
  TfLiteTensor* output_tensor = interpreter->tensor(output);
#endif
  const int width = output_tensor->dims->data[2];
  const int channels = output_tensor->dims->data[3];
  const int outputChannels = outputChannelsIn ? outputChannelsIn : channels;
  assert(outputChannels >= channels);
  if (widthOut) *widthOut = width;
  if (heightOut) *heightOut = width;
  
  NSMutableData *data = nil;
  if (output_tensor->type == kTfLiteUInt8) {
    int size = width*width*outputChannels;
    data = [[NSMutableData dataWithCapacity: size] initWithLength: size];
    uint8_t* out = (uint8_t*)[data bytes], *outEnd = out + width*width*outputChannels;
#ifdef TFLITE2
    const uint8_t* bytes = output_tensor->data.uint8;
#else
    const uint8_t* bytes = interpreter->typed_tensor<uint8_t>(output);
#endif
    while (out != outEnd) {
      for (int c = 0; c < channels; c++)
        *out++ = *bytes++;
      for (int c = 0; c < outputChannels - channels; c++)
        *out++ = 255;
    }
  } else { // kTfLiteFloat32
    if (convertToUint8) {
      int size = width*width*outputChannels;
      data = [[NSMutableData dataWithCapacity: size] initWithLength: size];
      uint8_t* out = (uint8_t*)[data bytes], *outEnd = out + width*width*outputChannels;
#ifdef TFLITE2
      const float* bytes = output_tensor->data.f;
#else
      const float* bytes = interpreter->typed_tensor<float>(output);
#endif
      while (out != outEnd) {
        for (int c = 0; c < channels; c++)
          *out++ = (*bytes++ * std) + mean;
        for (int c = 0; c < outputChannels - channels; c++)
          *out++ = 255;
      }
    } else { // kTfLiteFloat32
      int size = width*width*outputChannels*4;
      data = [[NSMutableData dataWithCapacity: size] initWithLength: size];
      float* out = (float*)[data bytes], *outEnd = out + width*width*outputChannels;
#ifdef TFLITE2
      float* bytes = output_tensor->data.f;
#else
      const float* bytes = interpreter->typed_tensor<float>(output);
#endif
      while (out != outEnd) {
        for (int c = 0; c < channels; c++)
          *out++ = (*bytes++ * std) + mean;
        for (int c = 0; c < outputChannels - channels; c++)
          *out++ = 255;
      }
    }
  }
  return data;
}

void feedInputTensorBinary(const FlutterStandardTypedData* typedData, int* input_size) {
#ifdef TFLITE2
  assert(TfLiteInterpreterGetInputTensorCount(interpreter) == 1);
  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
#else
  assert(interpreter->inputs().size() == 1);
  int input = interpreter->inputs()[0];
  TfLiteTensor* input_tensor = interpreter->tensor(input);
#endif
  const int width = input_tensor->dims->data[2];
  *input_size = width;
  NSData* in = [typedData data];

  if (input_tensor->type == kTfLiteUInt8) {
#ifdef TFLITE2
    TfLiteTensorCopyFromBuffer(input_tensor, in.bytes, in.length);
#else
    uint8_t* out = interpreter->typed_tensor<uint8_t>(input);
    const uint8_t* bytes = (const uint8_t*)[in bytes];
    for (int index = 0; index < [in length]; index++)
      out[index] = bytes[index];
#endif
  } else { // kTfLiteFloat32
#ifdef TFLITE2
    TfLiteTensorCopyFromBuffer(input_tensor, in.bytes, in.length);
#else
    float* out = interpreter->typed_tensor<float>(input);
    const float* bytes = (const float*)[in bytes];
    for (int index = 0; index < [in length]/4; index++)
      out[index] = bytes[index];
#endif
  }
}

void feedInputTensor(uint8_t* in, int* input_size, int image_height, int image_width, int image_channels, float input_mean, float input_std) {
#ifdef TFLITE2
  assert(TfLiteInterpreterGetInputTensorCount(interpreter) == 1);
  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
#else
  assert(interpreter->inputs().size() == 1);
  int input = interpreter->inputs()[0];
  TfLiteTensor* input_tensor = interpreter->tensor(input);
#endif
  const int input_channels = input_tensor->dims->data[3];
  const int width = input_tensor->dims->data[2];
  const int height = input_tensor->dims->data[1];
  *input_size = width;

  if (input_tensor->type == kTfLiteUInt8) {
#ifdef TFLITE2
    uint8_t* out = input_tensor->data.uint8;
#else
    uint8_t* out = interpreter->typed_tensor<uint8_t>(input);
#endif
    for (int y = 0; y < height; ++y) {
      const int in_y = (y * image_height) / height;
      uint8_t* in_row = in + (in_y * image_width * image_channels);
      uint8_t* out_row = out + (y * width * input_channels);
      for (int x = 0; x < width; ++x) {
        const int in_x = (x * image_width) / width;
        uint8_t* in_pixel = in_row + (in_x * image_channels);
        uint8_t* out_pixel = out_row + (x * input_channels);
        for (int c = 0; c < input_channels; ++c) {
          out_pixel[c] = in_pixel[c];
        }
      }
    }
  } else { // kTfLiteFloat32
#ifdef TFLITE2
    float* out = input_tensor->data.f;
#else
    float* out = interpreter->typed_tensor<float>(input);
#endif
    for (int y = 0; y < height; ++y) {
      const int in_y = (y * image_height) / height;
      uint8_t* in_row = in + (in_y * image_width * image_channels);
      float* out_row = out + (y * width * input_channels);
      for (int x = 0; x < width; ++x) {
        const int in_x = (x * image_width) / width;
        uint8_t* in_pixel = in_row + (in_x * image_channels);
        float* out_pixel = out_row + (x * input_channels);
        for (int c = 0; c < input_channels; ++c) {
          out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
        }
      }
    }
  }
}

void feedInputTensorImage(const NSString* image_path, float input_mean, float input_std, int* input_size) {
  int image_channels;
  int image_height;
  int image_width;
  std::vector<uint8_t> image_data = LoadImageFromFile([image_path UTF8String], &image_width, &image_height, &image_channels);
  uint8_t* in = image_data.data();
  feedInputTensor(in, input_size, image_height, image_width, image_channels, input_mean, input_std);
}

void feedInputTensorFrame(const FlutterStandardTypedData* typedData, int* input_size,
                          int image_height, int image_width, int image_channels, float input_mean, float input_std) {
  uint8_t* in = (uint8_t*)[[typedData data] bytes];
  feedInputTensor(in, input_size, image_height, image_width, image_channels, input_mean, input_std);
}

NSMutableArray* GetTopN(const float* prediction, const unsigned long prediction_size, const int num_results,
                    const float threshold) {
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
  std::greater<std::pair<float, int>>> top_result_pq;
  std::vector<std::pair<float, int>> top_results;
  
  const long count = prediction_size;
  for (int i = 0; i < count; ++i) {
    const float value = prediction[i];
    
    if (value < threshold) {
      continue;
    }
    
    top_result_pq.push(std::pair<float, int>(value, i));
    
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }
  }
  
  while (!top_result_pq.empty()) {
    top_results.push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results.begin(), top_results.end());
  
  NSMutableArray* predictions = [NSMutableArray array];
  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;
    NSString* labelObject = [NSString stringWithUTF8String:labels[index].c_str()];
    NSNumber* valueObject = [NSNumber numberWithFloat:confidence];
    NSMutableDictionary* res = [NSMutableDictionary dictionary];
    [res setValue:[NSNumber numberWithInt:index] forKey:@"index"];
    [res setObject:labelObject forKey:@"label"];
    [res setObject:valueObject forKey:@"confidence"];
    [predictions addObject:res];
  }
  
  return predictions;
}

void runModelOnImage(NSDictionary* args, FlutterResult result) {
  const NSString* image_path = args[@"path"];
  const float input_mean = [args[@"imageMean"] floatValue];
  const float input_std = [args[@"imageStd"] floatValue];
  
  NSMutableArray* empty = [@[] mutableCopy];
  
  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }
  
  int input_size;
  feedInputTensorImage(image_path, input_mean, input_std, &input_size);
  
  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }
    
#ifdef TFLITE2
    float* output = TfLiteInterpreterGetOutputTensor(interpreter, 0)->data.f;
#else
    float* output = interpreter->typed_output_tensor<float>(0);
#endif
    if (output == NULL)
      return result(empty);
    
    const unsigned long output_size = labels.size();
    const int num_results = [args[@"numResults"] intValue];
    const float threshold = [args[@"threshold"] floatValue];
    return result(GetTopN(output, output_size, num_results, threshold));
  });
}

void runModelOnBinary(NSDictionary* args, FlutterResult result) {
  const FlutterStandardTypedData* typedData = args[@"binary"];
  NSMutableArray* empty = [@[] mutableCopy];
  
  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }

  int input_size;
  feedInputTensorBinary(typedData, &input_size);
  
  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }
    
#ifdef TFLITE2
    float* output = TfLiteInterpreterGetOutputTensor(interpreter, 0)->data.f;
#else
    float* output = interpreter->typed_output_tensor<float>(0);
#endif
    if (output == NULL)
      return result(empty);
    
    const unsigned long output_size = labels.size();
    const int num_results = [args[@"numResults"] intValue];
    const float threshold = [args[@"threshold"] floatValue];
    return result(GetTopN(output, output_size, num_results, threshold));
  });
}

void runModelOnFrame(NSDictionary* args, FlutterResult result) {
  const FlutterStandardTypedData* typedData = args[@"bytesList"][0];
  const int image_height = [args[@"imageHeight"] intValue];
  const int image_width = [args[@"imageWidth"] intValue];
  const float input_mean = [args[@"imageMean"] floatValue];
  const float input_std = [args[@"imageStd"] floatValue];
  NSMutableArray* empty = [@[] mutableCopy];
  
  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }
  
  int input_size;
  int image_channels = 4;
  feedInputTensorFrame(typedData, &input_size, image_height, image_width, image_channels, input_mean, input_std);
  
  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }

#ifdef TFLITE2
    float* output = TfLiteInterpreterGetOutputTensor(interpreter, 0)->data.f;
#else
    float* output = interpreter->typed_output_tensor<float>(0);
#endif
    if (output == NULL)
      return result(empty);

    const unsigned long output_size = labels.size();
    const int num_results = [args[@"numResults"] intValue];
    const float threshold = [args[@"threshold"] floatValue];
    return result(GetTopN(output, output_size, num_results, threshold));
  });
}

NSMutableArray* parseSSDMobileNet(float threshold, int num_results_per_class) {
#ifdef TFLITE2
  assert(TfLiteInterpreterGetOutputTensorCount(interpreter) == 4);
#else
  assert(interpreter->outputs().size() == 4);
#endif
  NSMutableArray* results = [NSMutableArray array];
#ifdef TFLITE2
  float* output_locations = TfLiteInterpreterGetOutputTensor(interpreter, 0)->data.f;
  float* output_classes = TfLiteInterpreterGetOutputTensor(interpreter, 1)->data.f;
  float* output_scores = TfLiteInterpreterGetOutputTensor(interpreter, 2)->data.f;
  float* num_detections = TfLiteInterpreterGetOutputTensor(interpreter, 3)->data.f;
#else
  float* output_locations = interpreter->typed_output_tensor<float>(0);
  float* output_classes = interpreter->typed_output_tensor<float>(1);
  float* output_scores = interpreter->typed_output_tensor<float>(2);
  float* num_detections = interpreter->typed_output_tensor<float>(3);
#endif

  NSMutableDictionary* counters = [NSMutableDictionary dictionary];
  for (int d = 0; d < *num_detections; d++)
  {
    const int detected_class = output_classes[d];
    float score = output_scores[d];
    
    if (score < threshold) continue;
    
    NSMutableDictionary* res = [NSMutableDictionary dictionary];
    NSString* class_name = [NSString stringWithUTF8String:labels[detected_class + 1].c_str()];
    NSObject* counter = [counters objectForKey:class_name];
    
    if (counter) {
      int countValue = [(NSNumber*)counter intValue] + 1;
      if (countValue > num_results_per_class) {
        continue;
      }
      [counters setObject:@(countValue) forKey:class_name];
    } else {
      [counters setObject:@(1) forKey:class_name];
    }
    
    [res setObject:@(score) forKey:@"confidenceInClass"];
    [res setObject:class_name forKey:@"detectedClass"];
    
    const float ymin = fmax(0, output_locations[d * 4]);
    const float xmin = fmax(0, output_locations[d * 4 + 1]);
    const float ymax = output_locations[d * 4 + 2];
    const float xmax = output_locations[d * 4 + 3];

    NSMutableDictionary* rect = [NSMutableDictionary dictionary];
    [rect setObject:@(xmin) forKey:@"x"];
    [rect setObject:@(ymin) forKey:@"y"];
    [rect setObject:@(fmin(1 - xmin, xmax - xmin)) forKey:@"w"];
    [rect setObject:@(fmin(1 - ymin, ymax - ymin)) forKey:@"h"];
    
    [res setObject:rect forKey:@"rect"];
    [results addObject:res];
  }
  return results;
}

float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

void softmax(float vals[], int count) {
  float max = -FLT_MAX;
  for (int i=0; i<count; i++) {
    max = fmax(max, vals[i]);
  }
  float sum = 0.0;
  for (int i=0; i<count; i++) {
    vals[i] = exp(vals[i] - max);
    sum += vals[i];
  }
  for (int i=0; i<count; i++) {
    vals[i] /= sum;
  }
}

NSMutableArray* parseYOLO(int num_classes, const NSArray* anchors, int block_size, int num_boxes_per_bolock,
                          int num_results_per_class, float threshold, int input_size) {
#ifdef TFLITE2
  float* output = TfLiteInterpreterGetOutputTensor(interpreter, 0)->data.f;
#else
  float* output = interpreter->typed_output_tensor<float>(0);
#endif
  NSMutableArray* results = [NSMutableArray array];
  std::priority_queue<std::pair<float, NSMutableDictionary*>, std::vector<std::pair<float, NSMutableDictionary*>>,
  std::less<std::pair<float, NSMutableDictionary*>>> top_result_pq;
  
  int grid_size = input_size / block_size;
  for (int y = 0; y < grid_size; ++y) {
    for (int x = 0; x < grid_size; ++x) {
      for (int b = 0; b < num_boxes_per_bolock; ++b) {
        int offset = (grid_size * (num_boxes_per_bolock * (num_classes + 5))) * y
        + (num_boxes_per_bolock * (num_classes + 5)) * x
        + (num_classes + 5) * b;
        
        float confidence = sigmoid(output[offset + 4]);
        
        float classes[num_classes];
        for (int c = 0; c < num_classes; ++c) {
          classes[c] = output[offset + 5 + c];
        }
        
        softmax(classes, num_classes);
        
        int detected_class = -1;
        float max_class = 0;
        for (int c = 0; c < num_classes; ++c) {
          if (classes[c] > max_class) {
            detected_class = c;
            max_class = classes[c];
          }
        }
        
        float confidence_in_class = max_class * confidence;
        if (confidence_in_class > threshold) {
          NSMutableDictionary* rect = [NSMutableDictionary dictionary];
          NSMutableDictionary* res = [NSMutableDictionary dictionary];
          
          float xPos = (x + sigmoid(output[offset + 0])) * block_size;
          float yPos = (y + sigmoid(output[offset + 1])) * block_size;
          
          float anchor_w = [[anchors objectAtIndex:(2 * b + 0)] floatValue];
          float anchor_h = [[anchors objectAtIndex:(2 * b + 1)] floatValue];
          float w = (float) (exp(output[offset + 2]) * anchor_w) * block_size;
          float h = (float) (exp(output[offset + 3]) * anchor_h) * block_size;
          
          float x = fmax(0, (xPos - w / 2) / input_size);
          float y = fmax(0, (yPos - h / 2) / input_size);
          [rect setObject:@(x) forKey:@"x"];
          [rect setObject:@(y) forKey:@"y"];
          [rect setObject:@(fmin(1 - x, w / input_size)) forKey:@"w"];
          [rect setObject:@(fmin(1 - y, h / input_size)) forKey:@"h"];
          
          [res setObject:rect forKey:@"rect"];
          [res setObject:@(confidence_in_class) forKey:@"confidenceInClass"];
          NSString* class_name = [NSString stringWithUTF8String:labels[detected_class].c_str()];
          [res setObject:class_name forKey:@"detectedClass"];
          
          top_result_pq.push(std::pair<float, NSMutableDictionary*>(confidence_in_class, res));
        }
      }
    }
  }
  
  NSMutableDictionary* counters = [NSMutableDictionary dictionary];
  while (!top_result_pq.empty()) {
    NSMutableDictionary* result = top_result_pq.top().second;
    top_result_pq.pop();
    
    NSString* detected_class = [result objectForKey:@"detectedClass"];
    NSObject* counter = [counters objectForKey:detected_class];
    if (counter) {
      int countValue = [(NSNumber*)counter intValue] + 1;
      if (countValue > num_results_per_class) {
        continue;
      }
      [counters setObject:@(countValue) forKey:detected_class];
    } else {
      [counters setObject:@(1) forKey:detected_class];
    }
    [results addObject:result];
  }
  
  return results;
}

void detectObjectOnImage(NSDictionary* args, FlutterResult result) {
  const NSString* image_path = args[@"path"];
  const NSString* model = args[@"model"];
  const float threshold = [args[@"threshold"] floatValue];
  const float input_mean = [args[@"imageMean"] floatValue];
  const float input_std = [args[@"imageStd"] floatValue];
  const int num_results_per_class = [args[@"numResultsPerClass"] intValue];
  
  const NSArray* anchors = args[@"anchors"];
  const int num_boxes_per_block = [args[@"numBoxesPerBlock"] intValue];
  const int block_size = [args[@"blockSize"] floatValue];
  
  NSMutableArray* empty = [@[] mutableCopy];
  
  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }
  
  int input_size;
  feedInputTensorImage(image_path, input_mean, input_std, &input_size);
  
  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }

    if ([model isEqual: @"SSDMobileNet"])
      return result(parseSSDMobileNet(threshold, num_results_per_class));
    else
      return result(parseYOLO((int)labels.size(), anchors, block_size, num_boxes_per_block, num_results_per_class,
                              threshold, input_size));
  });
}

void detectObjectOnBinary(NSDictionary* args, FlutterResult result) {
  const FlutterStandardTypedData* typedData = args[@"binary"];
  const NSString* model = args[@"model"];
  const float threshold = [args[@"threshold"] floatValue];
  const int num_results_per_class = [args[@"numResultsPerClass"] intValue];
  
  const NSArray* anchors = args[@"anchors"];
  const int num_boxes_per_block = [args[@"numBoxesPerBlock"] intValue];
  const int block_size = [args[@"blockSize"] floatValue];
  
  NSMutableArray* empty = [@[] mutableCopy];
  
  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }
  
  int input_size;
  feedInputTensorBinary(typedData, &input_size);
  
  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }

    if ([model isEqual: @"SSDMobileNet"])
      return result(parseSSDMobileNet(threshold, num_results_per_class));
    else
      return result(parseYOLO((int)(labels.size() - 1), anchors, block_size, num_boxes_per_block, num_results_per_class,
                              threshold, input_size));
  });
}

void detectObjectOnFrame(NSDictionary* args, FlutterResult result) {
  const FlutterStandardTypedData* typedData = args[@"bytesList"][0];
  const NSString* model = args[@"model"];
  const int image_height = [args[@"imageHeight"] intValue];
  const int image_width = [args[@"imageWidth"] intValue];
  const float input_mean = [args[@"imageMean"] floatValue];
  const float input_std = [args[@"imageStd"] floatValue];
  const float threshold = [args[@"threshold"] floatValue];
  const int num_results_per_class = [args[@"numResultsPerClass"] intValue];
  
  const NSArray* anchors = args[@"anchors"];
  const int num_boxes_per_block = [args[@"numBoxesPerBlock"] intValue];
  const int block_size = [args[@"blockSize"] floatValue];
  
  NSMutableArray* empty = [@[] mutableCopy];
  
  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }
  
  int input_size;
  int image_channels = 4;
  feedInputTensorFrame(typedData, &input_size, image_height, image_width, image_channels, input_mean, input_std);
  
  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }

    if ([model isEqual: @"SSDMobileNet"])
      return result(parseSSDMobileNet(threshold, num_results_per_class));
    else
      return result(parseYOLO((int)labels.size(), anchors, block_size, num_boxes_per_block, num_results_per_class,
                              threshold, input_size));
  });
}

void runPix2PixOnImage(NSDictionary* args, FlutterResult result) {
  const NSString* image_path = args[@"path"];
  const float input_mean = [args[@"imageMean"] floatValue];
  const float input_std = [args[@"imageStd"] floatValue];
  const NSString* outputType = args[@"outputType"];
  NSMutableArray* empty = [@[] mutableCopy];

  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }

  int input_size;
  feedInputTensorImage(image_path, input_mean, input_std, &input_size);

  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }

    int width = 0, height = 0;
    NSMutableData* output = feedOutputTensor(4, input_mean, input_std, true, &width, &height);
    if (output == NULL)
      return result(empty);

    if ([outputType isEqual: @"png"]) {
      return result(CompressImage(output, width, height, 1));
    } else {
      return result(output);
    }
  });
}

void runPix2PixOnBinary(NSDictionary* args, FlutterResult result) {
  const FlutterStandardTypedData* typedData = args[@"binary"];
  NSMutableArray* empty = [@[] mutableCopy];

  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }

  int input_size;
  feedInputTensorBinary(typedData, &input_size);

  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }

    int width = 0, height = 0;
    NSMutableData* output = feedOutputTensor(0, 0, 1, false, &width, &height);
    if (output == NULL)
      return result(empty);

    return result(output);
  });
}

void runPix2PixOnFrame(NSDictionary* args, FlutterResult result) {
  const FlutterStandardTypedData* typedData = args[@"bytesList"][0];
  const int image_height = [args[@"imageHeight"] intValue];
  const int image_width = [args[@"imageWidth"] intValue];
  const float input_mean = [args[@"imageMean"] floatValue];
  const float input_std = [args[@"imageStd"] floatValue];
  const NSString* outputType = args[@"outputType"];
  NSMutableArray* empty = [@[] mutableCopy];

  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }

  int input_size;
  int image_channels = 4;
  feedInputTensorFrame(typedData, &input_size, image_height, image_width, image_channels, input_mean, input_std);

  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }

    int width = 0, height = 0;
    NSMutableData* output = feedOutputTensor(image_channels, input_mean, input_std, true, &width, &height);
    if (output == NULL)
      return result(empty);

    if ([outputType isEqual: @"png"]) {
      return result(CompressImage(output, width, height, 1));
    } else {
      return result(output);
    }
  });
}

void setPixel(char* rgba, int index, long color) {
  rgba[index * 4] = (color >> 16) & 0xFF;
  rgba[index * 4 + 1] = (color >> 8) & 0xFF;
  rgba[index * 4 + 2] = color & 0xFF;
  rgba[index * 4 + 3] = (color >> 24) & 0xFF;
}

NSData* fetchArgmax(const NSArray* labelColors, const NSString* outputType) {
#ifdef TFLITE2
  const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
#else
  int output = interpreter->outputs()[0];
  TfLiteTensor* output_tensor = interpreter->tensor(output);
#endif
  const int height = output_tensor->dims->data[1];
  const int width = output_tensor->dims->data[2];
  const int channels = output_tensor->dims->data[3];
  
  NSMutableData *data = nil;
  int size = height * width * 4;
  data = [[NSMutableData dataWithCapacity: size] initWithLength: size];
  char* out = (char*)[data bytes];
  if (output_tensor->type == kTfLiteUInt8) {
#ifdef TFLITE2
    const uint8_t* bytes = output_tensor->data.uint8;
#else
    const uint8_t* bytes = interpreter->typed_tensor<uint8_t>(output);
#endif
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        int index = i * width + j;
        int maxIndex = 0;
        int maxValue = 0;
        for (int c = 0; c < channels; ++c) {
          int outputValue = bytes[index* channels + c];
          if (outputValue > maxValue) {
            maxIndex = c;
            maxValue = outputValue;
          }
        }
        long labelColor = [[labelColors objectAtIndex:maxIndex] longValue];
        setPixel(out, index, labelColor);
      }
    }
  } else { // kTfLiteFloat32
#ifdef TFLITE2
    const float* bytes = output_tensor->data.f;
#else
    const float* bytes = interpreter->typed_tensor<float>(output);
#endif
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        int index = i * width + j;
        int maxIndex = 0;
        float maxValue = .0f;
        for (int c = 0; c < channels; ++c) {
          float outputValue = bytes[index * channels + c];
          if (outputValue > maxValue) {
            maxIndex = c;
            maxValue = outputValue;
          }
        }
        long labelColor = [[labelColors objectAtIndex:maxIndex] longValue];
        setPixel(out, index, labelColor);
      }
    }
  }
  
  if ([outputType isEqual: @"png"]) {
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef bitmapContext = CGBitmapContextCreate(out,
                                                       width,
                                                       height,
                                                       8, // bitsPerComponent
                                                       4 * width, // bytesPerRow
                                                       colorSpace,
                                                       kCGImageAlphaNoneSkipLast);
    
    CFRelease(colorSpace);
    CGImageRef cgImage = CGBitmapContextCreateImage(bitmapContext);
    NSData* image = UIImagePNGRepresentation([[UIImage alloc] initWithCGImage:cgImage]);
    CFRelease(cgImage);
    CFRelease(bitmapContext);
    return image;
  } else {
    return data;
  }
}

void runSegmentationOnImage(NSDictionary* args, FlutterResult result) {
  const NSString* image_path = args[@"path"];
  const float input_mean = [args[@"imageMean"] floatValue];
  const float input_std = [args[@"imageStd"] floatValue];
  const NSArray* labelColors = args[@"labelColors"];
  const NSString* outputType = args[@"outputType"];
  NSMutableArray* empty = [@[] mutableCopy];
  
  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }
  
  int input_size;
  feedInputTensorImage(image_path, input_mean, input_std, &input_size);
  
  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }

    NSData* output = fetchArgmax(labelColors, outputType);
    FlutterStandardTypedData* ret = [FlutterStandardTypedData typedDataWithBytes: output];
    return result(ret);
  });
}

void runSegmentationOnBinary(NSDictionary* args, FlutterResult result) {
  const FlutterStandardTypedData* typedData = args[@"binary"];
  const NSArray* labelColors = args[@"labelColors"];
  const NSString* outputType = args[@"outputType"];
  NSMutableArray* empty = [@[] mutableCopy];
  
  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }
  
  int input_size;
  feedInputTensorBinary(typedData, &input_size);
  
  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }

    NSData* output = fetchArgmax(labelColors, outputType);
    FlutterStandardTypedData* ret = [FlutterStandardTypedData typedDataWithBytes: output];
    return result(ret);
  });
}

void runSegmentationOnFrame(NSDictionary* args, FlutterResult result) {
  const FlutterStandardTypedData* typedData = args[@"bytesList"][0];
  const int image_height = [args[@"imageHeight"] intValue];
  const int image_width = [args[@"imageWidth"] intValue];
  const float input_mean = [args[@"imageMean"] floatValue];
  const float input_std = [args[@"imageStd"] floatValue];
  const NSArray* labelColors = args[@"labelColors"];
  const NSString* outputType = args[@"outputType"];
  NSMutableArray* empty = [@[] mutableCopy];

  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }
  
  int input_size;
  int image_channels = 4;
  feedInputTensorFrame(typedData, &input_size, image_height, image_width, image_channels, input_mean, input_std);
  
  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }

    NSData* output = fetchArgmax(labelColors, outputType);
    FlutterStandardTypedData* ret = [FlutterStandardTypedData typedDataWithBytes: output];
    return result(ret);
  });
}


NSArray* _tflite_part_names = @[
  @"nose", @"leftEye", @"rightEye", @"leftEar", @"rightEar", @"leftShoulder",
  @"rightShoulder", @"leftElbow", @"rightElbow", @"leftWrist", @"rightWrist",
  @"leftHip", @"rightHip", @"leftKnee", @"rightKnee", @"leftAnkle", @"rightAnkle"
];

NSArray* _tflite_pose_chain = @[
   @[@"nose", @"leftEye"], @[@"leftEye", @"leftEar"], @[@"nose", @"rightEye"],
   @[@"rightEye", @"rightEar"], @[@"nose", @"leftShoulder"],
   @[@"leftShoulder", @"leftElbow"], @[@"leftElbow", @"leftWrist"],
   @[@"leftShoulder", @"leftHip"], @[@"leftHip", @"leftKnee"],
   @[@"leftKnee", @"leftAnkle"], @[@"nose", @"rightShoulder"],
   @[@"rightShoulder", @"rightElbow"], @[@"rightElbow", @"rightWrist"],
   @[@"rightShoulder", @"rightHip"], @[@"rightHip", @"rightKnee"],
   @[@"rightKnee", @"rightAnkle"]
];

NSMutableDictionary* _tflite_parts_ids = [NSMutableDictionary dictionary];
NSMutableArray* _tflite_parent_to_child_edges = [NSMutableArray array];
NSMutableArray* _tflite_child_to_parent_edges = [NSMutableArray array];
int _tflite_local_maximum_radius = 1;
int _tflite_output_stride = 16;
int _tflite_height;
int _tflite_width;
int _tflite_num_keypoints;

void initPoseNet() {
  if ([_tflite_parts_ids count] == 0) {
    for (int i = 0; i < [_tflite_part_names count]; ++i)
      [_tflite_parts_ids setValue:[NSNumber numberWithInt:i] forKey:_tflite_part_names[i]];
    
    for (int i = 0; i < [_tflite_pose_chain count]; ++i) {
      [_tflite_parent_to_child_edges addObject:_tflite_parts_ids[_tflite_pose_chain[i][1]]];
      [_tflite_child_to_parent_edges addObject:_tflite_parts_ids[_tflite_pose_chain[i][0]]];
    }
  }
}

bool scoreIsMaximumInLocalWindow(int keypoint_id,
                                 float score,
                                 int heatmap_y,
                                 int heatmap_x,
                                 int local_maximum_radius,
                                 float* scores) {
  bool local_maxium = true;
  
  int y_start = MAX(heatmap_y - local_maximum_radius, 0);
  int y_end = MIN(heatmap_y + local_maximum_radius + 1, _tflite_height);
  for (int y_current = y_start; y_current < y_end; ++y_current) {
    int x_start = MAX(heatmap_x - local_maximum_radius, 0);
    int x_end = MIN(heatmap_x + local_maximum_radius + 1, _tflite_width);
    for (int x_current = x_start; x_current < x_end; ++x_current) {
      if (sigmoid(scores[(y_current * _tflite_width + x_current) * _tflite_num_keypoints + keypoint_id]) > score) {
        local_maxium = false;
        break;
      }
    }
    if (!local_maxium) {
      break;
    }
  }
  return local_maxium;
}

typedef std::priority_queue<std::pair<float, NSMutableDictionary*>,
std::vector<std::pair<float, NSMutableDictionary*>>,
std::less<std::pair<float, NSMutableDictionary*>>> PriorityQueue;

PriorityQueue buildPartWithScoreQueue(float* scores,
                                      float threshold,
                                      int local_maximum_radius) {
  PriorityQueue pq;
  for (int heatmap_y = 0; heatmap_y < _tflite_height; ++heatmap_y) {
    for (int heatmap_x = 0; heatmap_x < _tflite_width; ++heatmap_x) {
      for (int keypoint_id = 0; keypoint_id < _tflite_num_keypoints; ++keypoint_id) {
        float score = sigmoid(scores[(heatmap_y * _tflite_width + heatmap_x) *
                                     _tflite_num_keypoints + keypoint_id]);
        if (score < threshold) continue;
        
        if (scoreIsMaximumInLocalWindow(keypoint_id, score, heatmap_y, heatmap_x,
                                        local_maximum_radius, scores)) {
          NSMutableDictionary* res = [NSMutableDictionary dictionary];
          [res setValue:[NSNumber numberWithFloat:score] forKey:@"score"];
          [res setValue:[NSNumber numberWithInt:heatmap_y] forKey:@"y"];
          [res setValue:[NSNumber numberWithInt:heatmap_x] forKey:@"x"];
          [res setValue:[NSNumber numberWithInt:keypoint_id] forKey:@"partId"];
          pq.push(std::pair<float, NSMutableDictionary*>(score, res));
        }
      }
    }
  }
  return pq;
}

void getImageCoords(float* res,
                    NSMutableDictionary* keypoint,
                    float* offsets) {
  int heatmap_y = [keypoint[@"y"] intValue];
  int heatmap_x = [keypoint[@"x"] intValue];
  int keypoint_id = [keypoint[@"partId"] intValue];
  
  int offset = (heatmap_y * _tflite_width + heatmap_x) * _tflite_num_keypoints * 2 + keypoint_id;
  float offset_y = offsets[offset];
  float offset_x = offsets[offset + _tflite_num_keypoints];
  res[0] = heatmap_y * _tflite_output_stride + offset_y;
  res[1] = heatmap_x * _tflite_output_stride + offset_x;
}


bool withinNmsRadiusOfCorrespondingPoint(NSMutableArray* poses,
                                         float squared_nms_radius,
                                         float y,
                                         float x,
                                         int keypoint_id,
                                         int input_size) {
  for (NSMutableDictionary* pose in poses) {
    NSMutableDictionary* keypoints = pose[@"keypoints"];
    NSMutableDictionary* correspondingKeypoint = keypoints[[NSNumber numberWithInt:keypoint_id]];
    float _x = [correspondingKeypoint[@"x"] floatValue] * input_size - x;
    float _y = [correspondingKeypoint[@"y"] floatValue] * input_size - y;
    float squaredDistance = _x * _x + _y * _y;
    if (squaredDistance <= squared_nms_radius)
      return true;
  }
  return false;
}

void getStridedIndexNearPoint(int* res, float _y, float _x) {
  int y_ = round(_y / _tflite_output_stride);
  int x_ = round(_x / _tflite_output_stride);
  int y = y_ < 0 ? 0 : y_ > _tflite_height - 1 ? _tflite_height - 1 : y_;
  int x = x_ < 0 ? 0 : x_ > _tflite_width - 1 ? _tflite_width - 1 : x_;
  res[0] = y;
  res[1] = x;
}

void getDisplacement(float* res, int edgeId, int* keypoint, float* displacements) {
  int num_edges = (int)[_tflite_parent_to_child_edges count];
  int y = keypoint[0];
  int x = keypoint[1];
  int offset = (y * _tflite_width + x) * num_edges * 2 + edgeId;
  res[0] = displacements[offset];
  res[1] = displacements[offset + num_edges];
}

float getInstanceScore(NSMutableDictionary* keypoints) {
  float scores = 0;
  for (NSMutableDictionary* keypoint in keypoints.allValues)
    scores += [keypoint[@"score"] floatValue];
  return scores / _tflite_num_keypoints;
}

NSMutableDictionary* traverseToTargetKeypoint(int edge_id,
                                              NSMutableDictionary* source_keypoint,
                                              int target_keypoint_id,
                                              float* scores,
                                              float* offsets,
                                              float* displacements,
                                              int input_size) {
  float source_keypoint_y = [source_keypoint[@"y"] floatValue] * input_size;
  float source_keypoint_x = [source_keypoint[@"x"] floatValue] * input_size;
  
  int source_keypoint_indices[2];
  getStridedIndexNearPoint(source_keypoint_indices, source_keypoint_y, source_keypoint_x);
  
  float displacement[2];
  getDisplacement(displacement, edge_id, source_keypoint_indices, displacements);
  
  float displaced_point[2];
  displaced_point[0] = source_keypoint_y + displacement[0];
  displaced_point[1] = source_keypoint_x + displacement[1];
  
  float* target_keypoint = displaced_point;
  
  int offset_refine_step = 2;
  for (int i = 0; i < offset_refine_step; i++) {
    int target_keypoint_indices[2];
    getStridedIndexNearPoint(target_keypoint_indices, target_keypoint[0], target_keypoint[1]);
    
    int target_keypoint_y = target_keypoint_indices[0];
    int target_keypoint_x = target_keypoint_indices[1];
    
    int offset = (target_keypoint_y * _tflite_width + target_keypoint_x) * _tflite_num_keypoints * 2 + target_keypoint_id;
    float offset_y = offsets[offset];
    float offset_x = offsets[offset + _tflite_num_keypoints];
    
    target_keypoint[0] = target_keypoint_y * _tflite_output_stride + offset_y;
    target_keypoint[1] = target_keypoint_x * _tflite_output_stride + offset_x;
  }
  
  int target_keypoint_indices[2];
  getStridedIndexNearPoint(target_keypoint_indices, target_keypoint[0], target_keypoint[1]);
  
  float score = sigmoid(scores[(target_keypoint_indices[0] * _tflite_width +
                                target_keypoint_indices[1]) * _tflite_num_keypoints + target_keypoint_id]);
  
  NSMutableDictionary* keypoint = [NSMutableDictionary dictionary];
  [keypoint setValue:[NSNumber numberWithFloat:score] forKey:@"score"];
  [keypoint setValue:[NSNumber numberWithFloat:target_keypoint[0] / input_size] forKey:@"y"];
  [keypoint setValue:[NSNumber numberWithFloat:target_keypoint[1] / input_size] forKey:@"x"];
  [keypoint setValue:_tflite_part_names[target_keypoint_id] forKey:@"part"];
  return keypoint;
}

NSMutableArray* parsePoseNet(int num_results, float threshold, int nms_radius, int input_size) {
  initPoseNet();

#ifdef TFLITE2
  assert(TfLiteInterpreterGetOutputTensorCount(interpreter) == 4);
  const TfLiteTensor* scores_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
#else
  assert(interpreter->outputs().size() == 4);
  TfLiteTensor* scores_tensor = interpreter->tensor(interpreter->outputs()[0]);
#endif
  _tflite_height = scores_tensor->dims->data[1];
  _tflite_width = scores_tensor->dims->data[2];
  _tflite_num_keypoints = scores_tensor->dims->data[3];

#ifdef TFLITE2
  float* scores = TfLiteInterpreterGetOutputTensor(interpreter, 0)->data.f;
  float* offsets = TfLiteInterpreterGetOutputTensor(interpreter, 1)->data.f;
  float* displacements_fwd = TfLiteInterpreterGetOutputTensor(interpreter, 2)->data.f;
  float* displacements_bwd = TfLiteInterpreterGetOutputTensor(interpreter, 3)->data.f;
#else
  float* scores = interpreter->typed_output_tensor<float>(0);
  float* offsets = interpreter->typed_output_tensor<float>(1);
  float* displacements_fwd = interpreter->typed_output_tensor<float>(2);
  float* displacements_bwd = interpreter->typed_output_tensor<float>(3);
#endif
  PriorityQueue pq = buildPartWithScoreQueue(scores, threshold, _tflite_local_maximum_radius);
  
  int num_edges = (int)[_tflite_parent_to_child_edges count];
  int sqared_nms_radius = nms_radius * nms_radius;
  
  NSMutableArray* results = [NSMutableArray array];
  
  while([results count] < num_results && !pq.empty()) {
    NSMutableDictionary* root = pq.top().second;
    pq.pop();
    
    float root_point[2];
    getImageCoords(root_point, root, offsets);
    
    if (withinNmsRadiusOfCorrespondingPoint(results, sqared_nms_radius, root_point[0], root_point[1],
                                            [root[@"partId"] intValue], input_size))
      continue;
    
    NSMutableDictionary* keypoint = [NSMutableDictionary dictionary];
    [keypoint setValue:[NSNumber numberWithFloat:[root[@"score"] floatValue]] forKey:@"score"];
    [keypoint setValue:[NSNumber numberWithFloat:root_point[0] / input_size] forKey:@"y"];
    [keypoint setValue:[NSNumber numberWithFloat:root_point[1] / input_size] forKey:@"x"];
    [keypoint setValue:_tflite_part_names[[root[@"partId"] intValue]] forKey:@"part"];
    
    NSMutableDictionary* keypoints = [NSMutableDictionary dictionary];
    [keypoints setObject:keypoint forKey:root[@"partId"]];
    
    for (int edge = num_edges - 1; edge >= 0; --edge) {
      int source_keypoint_id = [_tflite_parent_to_child_edges[edge] intValue];
      int target_keypoint_id = [_tflite_child_to_parent_edges[edge] intValue];
      if (keypoints[[NSNumber numberWithInt:source_keypoint_id]] &&
          !(keypoints[[NSNumber numberWithInt:target_keypoint_id]])) {
        keypoint = traverseToTargetKeypoint(edge, keypoints[[NSNumber numberWithInt:source_keypoint_id]],
                                            target_keypoint_id, scores, offsets, displacements_bwd, input_size);
        [keypoints setObject:keypoint forKey:[NSNumber numberWithInt:target_keypoint_id]];
      }
    }
    
    for (int edge = 0; edge < num_edges; ++edge) {
      int source_keypoint_id = [_tflite_child_to_parent_edges[edge] intValue];
      int target_keypoint_id = [_tflite_parent_to_child_edges[edge] intValue];
      if (keypoints[[NSNumber numberWithInt:source_keypoint_id]] &&
          !(keypoints[[NSNumber numberWithInt:target_keypoint_id]])) {
        keypoint = traverseToTargetKeypoint(edge, keypoints[[NSNumber numberWithInt:source_keypoint_id]],
                                            target_keypoint_id, scores, offsets, displacements_fwd, input_size);
        [keypoints setObject:keypoint forKey:[NSNumber numberWithInt:target_keypoint_id]];
      }
    }
    
    NSMutableDictionary* result = [NSMutableDictionary dictionary];
    [result setObject:keypoints forKey:@"keypoints"];
    [result setValue:[NSNumber numberWithFloat:getInstanceScore(keypoints)] forKey:@"score"];
    [results addObject:result];
  }
  
  return results;
}

void runPoseNetOnImage(NSDictionary* args, FlutterResult result) {
  const NSString* image_path = args[@"path"];
  const float input_mean = [args[@"imageMean"] floatValue];
  const float input_std = [args[@"imageStd"] floatValue];
  const int num_results = [args[@"numResults"] intValue];
  const float threshold = [args[@"threshold"] floatValue];
  const int nms_radius = [args[@"nmsRadius"] intValue];;
  NSMutableArray* empty = [@[] mutableCopy];
  
  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }
  
  int input_size;
  feedInputTensorImage(image_path, input_mean, input_std, &input_size);
  
  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }
    
    return result(parsePoseNet(num_results, threshold, nms_radius, input_size));
  });
}

void runPoseNetOnBinary(NSDictionary* args, FlutterResult result) {
  const FlutterStandardTypedData* typedData = args[@"binary"];
  const int num_results = [args[@"numResults"] intValue];
  const float threshold = [args[@"threshold"] floatValue];
  const int nms_radius = [args[@"nmsRadius"] intValue];;
  NSMutableArray* empty = [@[] mutableCopy];
  
  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }
  
  int input_size;
  feedInputTensorBinary(typedData, &input_size);
  
  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }
    
    return result(parsePoseNet(num_results, threshold, nms_radius, input_size));
  });
}

void runPoseNetOnFrame(NSDictionary* args, FlutterResult result) {
  const FlutterStandardTypedData* typedData = args[@"bytesList"][0];
  const int image_height = [args[@"imageHeight"] intValue];
  const int image_width = [args[@"imageWidth"] intValue];
  const float input_mean = [args[@"imageMean"] floatValue];
  const float input_std = [args[@"imageStd"] floatValue];
  const int num_results = [args[@"numResults"] intValue];
  const float threshold = [args[@"threshold"] floatValue];
  const int nms_radius = [args[@"nmsRadius"] intValue];;
  NSMutableArray* empty = [@[] mutableCopy];
  
  if (!interpreter || interpreter_busy) {
    NSLog(@"Failed to construct interpreter or busy.");
    return result(empty);
  }
  
  int input_size;
  int image_channels = 4;
  feedInputTensorFrame(typedData, &input_size, image_height, image_width, image_channels, input_mean, input_std);
  
  runTflite(args, ^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }
    
    return result(parsePoseNet(num_results, threshold, nms_radius, input_size));
  });
}

void close() {
#ifdef TFLITE2
  interpreter = nullptr;
  if (delegate != nullptr)
    TFLGpuDelegateDelete(delegate);
  delegate = nullptr;
#else
  interpreter.release();
  interpreter = NULL;
#endif
  model = NULL;
  labels.clear();
}

