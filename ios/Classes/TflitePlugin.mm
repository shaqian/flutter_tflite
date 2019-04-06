// #define CONTRIB_PATH

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
  } else if ([@"close" isEqualToString:call.method]) {
    close();
  } else {
    result(FlutterMethodNotImplemented);
  }
}

@end

std::vector<std::string> labels;
std::unique_ptr<tflite::FlatBufferModel> model;
std::unique_ptr<tflite::Interpreter> interpreter;
bool interpreter_busy = false;

static void LoadLabels(NSString* labels_path,
                       std::vector<std::string>* label_strings) {
  if (!labels_path) {
    LOG(ERROR) << "Failed to find label file at" << labels_path;
  }
  std::ifstream t;
  t.open([labels_path UTF8String]);
  std::string line;
  while (t) {
    std::getline(t, line);
    label_strings->push_back(line);
  }
  t.close();
}

NSString* loadModel(NSObject<FlutterPluginRegistrar>* _registrar, NSDictionary* args) {
  NSString* key = [_registrar lookupKeyForAsset:args[@"model"]];
  NSString* graph_path = [[NSBundle mainBundle] pathForResource:key ofType:nil];
  const int num_threads = [args[@"numThreads"] intValue];
  
  model = tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]);
  if (!model) {
    return [NSString stringWithFormat:@"%s %@", "Failed to mmap model", graph_path];
  }
  LOG(INFO) << "Loaded model " << graph_path;
  model->error_reporter();
  LOG(INFO) << "resolved reporter";
  
  key = [_registrar lookupKeyForAsset:args[@"labels"]];
  NSString* labels_path = [[NSBundle mainBundle] pathForResource:key ofType:nil];
  LoadLabels(labels_path, &labels);
  
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
  return @"success";
}

void runTflite(NSDictionary* args, TfLiteStatusCallback cb) {
  const bool asynch = [args[@"asynch"] boolValue];
  if (asynch) {
    interpreter_busy = true;
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(void){
      TfLiteStatus status = interpreter->Invoke();
      dispatch_async(dispatch_get_main_queue(), ^(void){
        interpreter_busy = false;
        cb(status);
      });
    });
  } else {
    TfLiteStatus status = interpreter->Invoke();
    cb(status);
  }
}

NSMutableData *feedOutputTensor(int outputChannelsIn, float mean, float std, bool convertToUint8,
                                int *widthOut, int *heightOut) {
  assert(interpreter->outputs().size() == 1);
  int output = interpreter->outputs()[0];
  TfLiteTensor* output_tensor = interpreter->tensor(output);
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
    const uint8_t* bytes = interpreter->typed_tensor<uint8_t>(output);
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
      const float* bytes = interpreter->typed_tensor<float>(output);
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
      const float* bytes = interpreter->typed_tensor<float>(output);
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
  assert(interpreter->inputs().size() == 1);
  int input = interpreter->inputs()[0];
  TfLiteTensor* input_tensor = interpreter->tensor(input);
  const int width = input_tensor->dims->data[2];
  *input_size = width;
  NSData* in = [typedData data];
  
  if (input_tensor->type == kTfLiteUInt8) {
    uint8_t* out = interpreter->typed_tensor<uint8_t>(input);
    const uint8_t* bytes = (const uint8_t*)[in bytes];
    for (int index = 0; index < [in length]; index++)
      out[index] = bytes[index];
  } else { // kTfLiteFloat32
    float* out = interpreter->typed_tensor<float>(input);
    const float* bytes = (const float*)[in bytes];
    for (int index = 0; index < [in length]/4; index++)
      out[index] = bytes[index];
  }
}

void feedInputTensor(uint8_t* in, int* input_size, int image_height, int image_width, int image_channels, float input_mean, float input_std) {
  assert(interpreter->inputs().size() == 1);
  int input = interpreter->inputs()[0];
  TfLiteTensor* input_tensor = interpreter->tensor(input);
  const int input_channels = input_tensor->dims->data[3];
  const int width = input_tensor->dims->data[2];
  const int height = input_tensor->dims->data[1];
  *input_size = width;
  
  if (input_tensor->type == kTfLiteUInt8) {
    uint8_t* out = interpreter->typed_tensor<uint8_t>(input);
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
    float* out = interpreter->typed_tensor<float>(input);
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

    float* output = interpreter->typed_output_tensor<float>(0);

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

    float* output = interpreter->typed_output_tensor<float>(0);

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

    float* output = interpreter->typed_output_tensor<float>(0);

    if (output == NULL)
      return result(empty);

    const unsigned long output_size = labels.size();
    const int num_results = [args[@"numResults"] intValue];
    const float threshold = [args[@"threshold"] floatValue];
    return result(GetTopN(output, output_size, num_results, threshold));
  });
}

NSMutableArray* parseSSDMobileNet(float threshold, int num_results_per_class) {
  assert(interpreter->outputs().size() == 4);
  
  NSMutableArray* results = [NSMutableArray array];
  float* output_locations = interpreter->typed_output_tensor<float>(0);
  float* output_classes = interpreter->typed_output_tensor<float>(1);
  float* output_scores = interpreter->typed_output_tensor<float>(2);
  float* num_detections = interpreter->typed_output_tensor<float>(3);
  
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
  float* output = interpreter->typed_output_tensor<float>(0);
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
      return result(parseYOLO((int)(labels.size() - 1), anchors, block_size, num_boxes_per_block, num_results_per_class,
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
      return result(parseYOLO((int)(labels.size() - 1), anchors, block_size, num_boxes_per_block, num_results_per_class,
                              threshold, input_size));
  });
}

void runPix2PixOnImage(NSDictionary* args, FlutterResult result) {
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

    int width = 0, height = 0;
    NSMutableData* output = feedOutputTensor(4, input_mean, input_std, true, &width, &height);
    if (output == NULL)
      return result(empty);

    NSString *ext = image_path.pathExtension, *out_path = image_path.stringByDeletingPathExtension;
    out_path = [NSString stringWithFormat:@"%@_pix2pix.%@", out_path, ext];
    if (SaveImageToFile(output, [out_path UTF8String], width, height, 1)) {
      NSMutableArray* results = [NSMutableArray array];
      NSMutableDictionary* res = [NSMutableDictionary dictionary];
      [res setObject:out_path forKey:@"filename"];
      [results addObject:res];
      return result(results);
    }

    return result(empty);
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

    FlutterStandardTypedData* ret = [FlutterStandardTypedData typedDataWithBytes: output];
    NSMutableArray* results = [NSMutableArray array];
    NSMutableDictionary* res = [NSMutableDictionary dictionary];
    [res setObject:ret forKey:@"binary"];
    [results addObject:res];
    return result(results);
  });
}

void runPix2PixOnFrame(NSDictionary* args, FlutterResult result) {
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

    int width = 0, height = 0;
    NSMutableData* output = feedOutputTensor(0, 0, 1, false, &width, &height);
    if (output == NULL)
      return result(empty);

    FlutterStandardTypedData* ret = [FlutterStandardTypedData typedDataWithBytes: output];
    NSMutableArray* results = [NSMutableArray array];
    NSMutableDictionary* res = [NSMutableDictionary dictionary];
    [res setObject:ret forKey:@"binary"];
    [results addObject:res];
    return result(results);
  });
}

void setPixel(char* rgba, int index, long color) {
  rgba[index * 4] = (color >> 16) & 0xFF;
  rgba[index * 4 + 1] = (color >> 8) & 0xFF;
  rgba[index * 4 + 2] = color & 0xFF;
  rgba[index * 4 + 3] = (color >> 24) & 0xFF;
}

NSData* fetchArgmax(const NSArray* labelColors, const NSString* outputType) {
  int output = interpreter->outputs()[0];
  TfLiteTensor* output_tensor = interpreter->tensor(output);
  const int height = output_tensor->dims->data[1];
  const int width = output_tensor->dims->data[2];
  const int channels = output_tensor->dims->data[3];
  
  NSMutableData *data = nil;
  int size = height * width * 4;
  data = [[NSMutableData dataWithCapacity: size] initWithLength: size];
  char* out = (char*)[data bytes];
  if (output_tensor->type == kTfLiteUInt8) {
    const uint8_t* bytes = interpreter->typed_tensor<uint8_t>(output);
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
    const float* bytes = interpreter->typed_tensor<float>(output);
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
  
  if (!interpreter) {
    NSLog(@"Failed to construct interpreter.");
    return result(empty);
  }
  
  int input_size;
  feedInputTensorImage(image_path, input_mean, input_std, &input_size);
  
  runTfliteAsync(^(TfLiteStatus status) {
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
  
  if (!interpreter) {
    NSLog(@"Failed to construct interpreter.");
    return result(empty);
  }
  
  int input_size;
  feedInputTensorBinary(typedData, &input_size);
  
  runTfliteAsync(^(TfLiteStatus status) {
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
  
  if (!interpreter) {
    NSLog(@"Failed to construct interpreter.");
    return result(empty);
  }
  
  int input_size;
  int image_channels = 4;
  feedInputTensorFrame(typedData, &input_size, image_height, image_width, image_channels, input_mean, input_std);
  
  runTfliteAsync(^(TfLiteStatus status) {
    if (status != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      return result(empty);
    }

    NSData* output = fetchArgmax(labelColors, outputType);
    FlutterStandardTypedData* ret = [FlutterStandardTypedData typedDataWithBytes: output];
    return result(ret);
  });
}

void close() {
  interpreter.release();
  interpreter = NULL;
  model = NULL;
  labels.clear();
}

