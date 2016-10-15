project(feature_processor)

find_package(OpenCV REQUIRED)

add_library(feature_processor
            feature_processor.h
            src/feature_processor.cpp)

target_link_libraries(feature_processor
                      ${OpenCV_LIBS})


add_executable(feature_processor_demo demo/feature_processor_demo.cpp)

target_link_libraries(feature_processor_demo
                      feature_processor
                      ${OpenCV_LIBS})
