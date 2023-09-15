cmake_minimum_required(VERSION 3.13)
project(TrtNet)
set(CMAKE_CXX_STANDARD 17)

# Add dependencies
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# add TensorRT8
set(TENSORRT_LIB_PATH /home/avent/.CppLib/TensorRT-8.5.3.1/lib)
file(GLOB TENSORRT_LIBS ${TENSORRT_LIB_PATH}/*.so)

# Add library
file(GLOB API_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} STATIC ${API_SOURCE_FILES}) # Or SHARED

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /home/avent/.CppLib/TensorRT-8.5.3.1/include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
    ${TENSORRT_LIBS}
    ${CUDA_LIBRARIES}
)
