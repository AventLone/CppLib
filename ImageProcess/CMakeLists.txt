cmake_minimum_required(VERSION 3.13)
project(ImageProcess)
set(CMAKE_CXX_STANDARD 17)

# Add sub-directory
# set(UTILS_LIB api)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils/parseSetting)

# add OpenCV
find_package(OpenCV REQUIRED)

file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
# add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

add_executable(test test/test.cpp)
target_include_directories(test PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/parseSetting/include
)
target_link_libraries(test ${PROJECT_NAME})
