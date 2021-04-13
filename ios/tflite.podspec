#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html
#
Pod::Spec.new do |s|
  s.name             = 'tflite'
  s.version          = '1.1.2'
  s.summary          = 'A Flutter plugin for accessing TensorFlow Lite.'
  s.description      = <<-DESC
A Flutter plugin for accessing TensorFlow Lite. Supports both iOS and Android.
                       DESC
  s.homepage         = 'https://github.com/shaqian/flutter_tflite'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Qian Sha' => 'https://github.com/shaqian' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.public_header_files = 'Classes/**/*.h'
  s.dependency 'Flutter'
  s.dependency 'TensorFlowLiteC'
  s.xcconfig = { 'USER_HEADER_SEARCH_PATHS' => '$(inherited) "${PODS_ROOT}/Headers/Private" "${PODS_ROOT}/Headers/Private/tflite" "${PODS_ROOT}/Headers/Public" "${PODS_ROOT}/Headers/Public/Flutter" "${PODS_ROOT}/Headers/Public/TensorFlowLite/tensorflow_lite" "${PODS_ROOT}/Headers/Public/tflite" "${PODS_ROOT}/TensorFlowLite/Frameworks/tensorflow_lite.framework/Headers" "${PODS_ROOT}/TensorFlowLiteC/Frameworks/TensorFlowLiteC.framework/Headers"' }

  s.ios.deployment_target = '9.0'
  s.static_framework = true
end

