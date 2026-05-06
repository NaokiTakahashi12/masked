set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
  include(FetchContent)
  FetchContent_Declare(googlebenchmark
    URL "https://github.com/google/benchmark/archive/afa23b7699c17f1e26c88cbf95257b20d78d6247.zip"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googlebenchmark)
endif()

add_executable(${PROJECT_NAME}_benchmark)
target_sources(${PROJECT_NAME}_benchmark
  PRIVATE
    benchmark_main.cpp
    benchmark_masked.cpp
)
target_link_libraries(${PROJECT_NAME}_benchmark
  PRIVATE
    benchmark::benchmark
    masked::masked
)
target_compile_options(${PROJECT_NAME}_benchmark
  PRIVATE
    ${masked_compile_options}
)
