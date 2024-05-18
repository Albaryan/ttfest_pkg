cmake_minimum_required(VERSION 3.5)
project(ttfest_pkg)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
include_directories(include)
include_directories(ttfest_pkg)

install (DIRECTORY include launch models scripts src ttfest_pkg worlds lane_following
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
	scripts/camera_node.py
	scripts/control_node.py
  scripts/lidar_node.py
  scripts/lane_following_node.py
	DESTINATION lib/${PROJECT_NAME}
)

ament_package()
