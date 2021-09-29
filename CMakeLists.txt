cmake_minimum_required(VERSION 3.0.2)
project(test_skeleton)

add_compile_options(-std=c++14 -funroll-loops -Wall -Ofast)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

find_package(catkin REQUIRED COMPONENTS
  graph_core
  graph_replanning
  roscpp
  object_loader_msgs
  moveit_planning_helper
  geometry_msgs
  configuration_msgs
  sound_play
  )

catkin_package(
  CATKIN_DEPENDS  graph_core graph_replanning roscpp object_loader_msgs moveit_planning_helper geometry_msgs configuration_msgs sound_play
  DEPENDS
  )
include_directories(
  ${catkin_INCLUDE_DIRS}
  )

add_executable(       test_planning src/test_planning.cpp)
add_dependencies(     test_planning ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(test_planning
  ${catkin_LIBRARIES}
  )
  add_executable(       test_scaling src/test_scaling.cpp)
  add_dependencies(     test_scaling ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(test_scaling
    ${catkin_LIBRARIES}
    )
