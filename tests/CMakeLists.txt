set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest QUIET)

if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL "https://github.com/google/googletest/archive/6910c9d9165801d8827d628cb72eb7ea9dd538c5.zip"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)
enable_testing()

add_executable(${PROJECT_NAME}_test)
target_sources(${PROJECT_NAME}_test
  PRIVATE
    test_masked.cpp
)
target_link_libraries(${PROJECT_NAME}_test
  PRIVATE
    GTest::gtest_main
    masked::masked
)
target_compile_options(${PROJECT_NAME}_test
  PRIVATE
    ${masked_compile_options}
)

gtest_discover_tests(${PROJECT_NAME}_test)
