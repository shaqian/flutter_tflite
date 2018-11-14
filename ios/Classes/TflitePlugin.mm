#import "TflitePlugin.h"

#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/op_resolver.h"

#include "ios_image_load.h"

#define LOG(x) std::cerr

NSString* loadModel(NSObject<FlutterPluginRegistrar>* _registrar, NSDictionary* args);
NSMutableArray* runModelOnImage(NSDictionary* args);
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
    NSMutableArray* inference_result = runModelOnImage(call.arguments);
    result(inference_result);
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
  
  return @"success";
}

static void GetTopN(const float* prediction, const unsigned long prediction_size, const int num_results,
                    const float threshold, std::vector<std::pair<float, int> >* top_results) {
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
  std::greater<std::pair<float, int> > >
  top_result_pq;
  
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
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}


NSMutableArray* runModelOnImage(NSDictionary* args) {
  const NSString* image_path = args[@"path"];
  const int num_threads = [args[@"numThreads"] intValue];
  const int wanted_width = [args[@"inputSize"] intValue];
  const int wanted_height = [args[@"inputSize"] intValue];
  const int wanted_channels = [args[@"numChannels"] intValue];
  const float input_mean = [args[@"imageMean"] floatValue];
  const float input_std = [args[@"imageStd"] floatValue];
  
  NSMutableArray* empty = [@[] mutableCopy];
  
  if (!interpreter) {
    NSLog(@"Failed to construct interpreter.");
    return empty;
  }
  
  if (num_threads != -1) {
    interpreter->SetNumThreads(num_threads);
  }
  
  int input = interpreter->inputs()[0];
  
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    NSLog(@"Failed to allocate tensors.");
    return empty;
  }
  
  int image_width;
  int image_height;
  int image_channels;
  std::vector<uint8_t> image_data = LoadImageFromFile([image_path UTF8String], &image_width, &image_height, &image_channels);
  
  assert(image_channels >= wanted_channels);
  uint8_t* in = image_data.data();
  float* out = interpreter->typed_tensor<float>(input);
  for (int y = 0; y < wanted_height; ++y) {
    const int in_y = (y * image_height) / wanted_height;
    uint8_t* in_row = in + (in_y * image_width * image_channels);
    float* out_row = out + (y * wanted_width * wanted_channels);
    for (int x = 0; x < wanted_width; ++x) {
      const int in_x = (x * image_width) / wanted_width;
      uint8_t* in_pixel = in_row + (in_x * image_channels);
      float* out_pixel = out_row + (x * wanted_channels);
      for (int c = 0; c < wanted_channels; ++c) {
        out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
      }
    }
  }
  
  if (interpreter->Invoke() != kTfLiteOk) {
    NSLog(@"Failed to invoke!");
    return empty;
  }
  
  float* output = interpreter->typed_output_tensor<float>(0);
  
  if (output == NULL)
    return empty;
  
  const unsigned long output_size = labels.size();
  const int kNumResults = [args[@"numResults"] intValue];
  const float kThreshold = [args[@"threshold"] floatValue];
  std::vector<std::pair<float, int> > top_results;
  GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
  
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

void close() {
  interpreter = NULL;
  model = NULL;
  labels.clear();
}
