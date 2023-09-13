cmake_minimum_required(VERSION 3.13)
project(ImageProcess)
set(CMAKE_CXX_STANDARD 17)
 
# add OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED
    src/FocusStacking.cpp
    src/Projector.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)