cmake_minimum_required(VERSION 3.16)
project(scheduling_solvers CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")
  set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -fsanitize=address,undefined")
endif()
if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
endif()

# All sources
set(SOURCES
    dp_solver.cpp
    bnb_solver.cpp
    compare.cpp
)

add_executable(compare ${SOURCES})

# Warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(compare PRIVATE -Wall -Wextra -Wpedantic)
endif()

# LTO disabled: adds 2-5min link time for ~10% runtime benefit, not worth it during dev.
# To re-enable: set_property(TARGET compare PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
