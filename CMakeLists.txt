cmake_minimum_required(VERSION 2.8)
project(imageRetieval)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(OpenCV_DIR "D:\\library\\opencv\\build")

find_package(OpenCV 3.1.0 EXACT REQUIRED PATHS OpenCV_DIR)
if(OpenCV_FOUND)
    message("${Red}-- Find OpenCV library 3.1.0${ColourReset}")
    message("-- OpenCV Dir:${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIBRARY_DIRS})
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(binfinal test/test.cpp src/akmeans.cpp)
target_link_libraries(binfinal ${OpenCV_LIBS})
