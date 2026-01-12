cmake_minimum_required(VERSION 3.28)
project(Tpool LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_library(tpool INTERFACE)

target_include_directories(tpool INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(tpool INTERFACE Threads::Threads)

# For GCC/Clang coroutines support.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(tpool INTERFACE -fcoroutines)
endif()

# Main.
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE tpool)
