cmake_minimum_required(VERSION 3.5)
project(ar_pkg)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}")



set(OpenCV_DIR "/usr/local/opencv-4.5.4/lib/cmake/opencv4")
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui aruco video)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ZED 4 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_typesupport_c REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)


ament_export_dependencies(rosidl_typesupport_c)
ament_export_dependencies(rosidl_typesupport_cpp)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArucoPose.msg"
  "msg/ArucoPoseArray.msg"
  "msg/Pos.msg"
  "msg/Ori.msg"
  "msg/Eul.msg"
  "msg/Quat.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/ar_pkg)
include_directories(${CMAKE_BINARY_DIR}/rosidl_generator_cpp)
include_directories(/usr/local/zed/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS}) 
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add library for custom ArUco code
add_library(aruco_lib src/aruco.cpp)
# Link OpenCV libraries to your custom ArUco library
target_link_libraries(aruco_lib ${OpenCV_LIBS})

link_directories(/usr/local/zed/lib)
link_directories(${ZED_LIBRARY_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

FILE(GLOB_RECURSE SRC_FILES src/*.cpp)
FILE(GLOB_RECURSE HDR_FILES include/*.hpp)

add_executable(Aruco_Node src/Aruco_Node.cpp)

ament_target_dependencies(Aruco_Node
  rclcpp
  geometry_msgs
  tf2
  tf2_ros
  tf2_msgs
)

target_link_libraries(Aruco_Node
  aruco_lib
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  sl_zed
  tf2_ros::tf2_ros
  ar_pkg__rosidl_typesupport_c
  ar_pkg__rosidl_typesupport_cpp
)

install(TARGETS
  Aruco_Node
	aruco_lib
	DESTINATION lib/${PROJECT_NAME})

ament_package()
