# Sets the minimum version of CMake required to build the native
# library. You should either keep the default value or only pass a
# value of 3.4.0 or lower.

cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


set(PATH_TO_PROJECT /Users/flb/Downloads/SmartObjectRecognition)
set(PATH_TO_OPENCV_SDK /Users/flb/Documents/SourceCode/OpenCV-2.4.10-android-sdk)

include_directories(${PATH_TO_OPENCV_SDK}/sdk/native/jni/include)



add_library( open-cv SHARED IMPORTED )
set_target_properties(open-cv PROPERTIES IMPORTED_LOCATION ${PATH_TO_PROJECT}/app/src/main/jniLibs/${ANDROID_ABI}/libopencv_java.so)

add_library( native-lib SHARED src/main/cpp/native-lib.cpp )

find_library( log-lib log )

target_link_libraries( native-lib $\{log-lib}  open-cv)

