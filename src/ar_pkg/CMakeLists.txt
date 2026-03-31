cmake_minimum_required(VERSION 3.5)
project(ar_pkg)

option(BUILD_ZED_NODE "Build the ZED-based node" ON)
option(BUILD_GENERIC_NODE "Build the generic webcam node" ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}")

set(OpenCV_DIR "/usr/local/opencv-4.5.4/lib/cmake/opencv4")
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui aruco video)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
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

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/ar_pkg
  ${CMAKE_BINARY_DIR}/rosidl_generator_cpp
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

# Shared non-ZED helper library only
add_library(aruco_lib src/aruco.cpp)
target_link_libraries(aruco_lib ${OpenCV_LIBS})

if(BUILD_GENERIC_NODE)
  add_executable(Aruco_Generic_Node src/Aruco_Generic_Node.cpp)

  ament_target_dependencies(Aruco_Generic_Node
    rclcpp
    geometry_msgs
    tf2
    tf2_ros
    tf2_msgs
  )

  target_link_libraries(Aruco_Generic_Node
    aruco_lib
    ${OpenCV_LIBS}
    tf2_ros::tf2_ros
    ar_pkg__rosidl_typesupport_c
    ar_pkg__rosidl_typesupport_cpp
  )

  install(TARGETS
    Aruco_Generic_Node
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

if(BUILD_ZED_NODE)
  find_package(ZED 4 REQUIRED)
  find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)

  include_directories(
    /usr/local/zed/include
    ${CUDA_INCLUDE_DIRS}
    ${ZED_INCLUDE_DIRS}
  )

  link_directories(
    /usr/local/zed/lib
    ${ZED_LIBRARY_DIR}
    ${CUDA_LIBRARY_DIRS}
  )

  add_executable(Aruco_Zed_Node src/Aruco_Zed_Node.cpp)

  ament_target_dependencies(Aruco_Zed_Node
    rclcpp
    geometry_msgs
    tf2
    tf2_ros
    tf2_msgs
  )

  target_link_libraries(Aruco_Zed_Node
    aruco_lib
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    sl_zed
    tf2_ros::tf2_ros
    ar_pkg__rosidl_typesupport_c
    ar_pkg__rosidl_typesupport_cpp
  )

  install(TARGETS
    Aruco_Zed_Node
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

install(TARGETS
  aruco_lib
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
