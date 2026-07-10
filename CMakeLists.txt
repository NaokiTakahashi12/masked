cmake_minimum_required(VERSION 3.18)

set(masked_MAJOR_VERSION 0)
set(masked_MINOR_VERSION 0)
set(masked_PATCH_VERSION 1)
set(masked_VERSION
  ${masked_MAJOR_VERSION}.${masked_MINOR_VERSION}.${masked_PATCH_VERSION})

project(masked
  VERSION ${masked_VERSION}
  LANGUAGES
    CXX
)

set(masked_compile_options)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(masked_compile_options
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wpointer-arith
    -Wcast-qual
  )
endif()

if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "CMake build type" FORCE)
  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY
      STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
endif()

option(MASKED_BUILD_DEMOS "Build masked demo executables" ${PROJECT_IS_TOP_LEVEL})
option(MASKED_BUILD_TESTING "Build masked tests and benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(MASKED_INSTALL "Enable install/export rules for masked" ${PROJECT_IS_TOP_LEVEL})

include(GNUInstallDirs)

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(${PROJECT_NAME}
  INTERFACE
    cxx_std_20
)
set_target_properties(${PROJECT_NAME}
  PROPERTIES
    EXPORT_NAME ${PROJECT_NAME}
)

if(MASKED_BUILD_DEMOS)
  add_executable(${PROJECT_NAME}_demo)
  target_sources(${PROJECT_NAME}_demo
    PRIVATE
      demos/masked_demo.cpp
  )
  target_link_libraries(${PROJECT_NAME}_demo
    PRIVATE
      ${PROJECT_NAME}::${PROJECT_NAME}
  )
  target_compile_options(${PROJECT_NAME}_demo
    PRIVATE
      ${masked_compile_options}
  )
  set_target_properties(${PROJECT_NAME}_demo
    PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
endif()

include(CMakePackageConfigHelpers)

if(MASKED_INSTALL)
  set(masked_install_cmakedir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${masked_install_cmakedir}
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
  )
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  install(
    TARGETS
      ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
  )
  install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  install(
    EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${masked_install_cmakedir}
  )
  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${masked_install_cmakedir}
  )
endif()

if(MASKED_BUILD_TESTING)
  include(CTest)
  if(BUILD_TESTING)
    add_subdirectory(tests)
    add_subdirectory(benchmarks)
  endif()
endif()

if(PROJECT_IS_TOP_LEVEL AND MASKED_INSTALL)
  include(cmake/Packaging.cmake)
endif()
