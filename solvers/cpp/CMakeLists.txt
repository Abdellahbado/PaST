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
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
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

add_executable(
  stateful_compare
  dp_solver.cpp
  stateful_dp_solver.cpp
  stateful_compare.cpp
)

add_executable(
  stateless_compare
  dp_solver.cpp
  stateful_dp_solver.cpp
  stateless_compare.cpp
)

add_executable(
  parallel_pattern_compare
  dp_solver.cpp
  stateful_dp_solver.cpp
  parallel_stateful_oracle.cpp
  parallel_pattern_compare.cpp
)

add_executable(
  pricing_compare
  dp_solver.cpp
  pricing_compare.cpp
)

add_executable(
  parallel_lbbd_compare
  dp_solver.cpp
  stateful_dp_solver.cpp
  parallel_stateful_oracle.cpp
  parallel_lbbd_compare.cpp
)

add_executable(
  parallel_frontier_runner
  parallel_frontier_runner.cpp
)

add_executable(
  parallel_campaign_runner
  parallel_campaign_runner.cpp
)

add_executable(
  parallel_config_master_compare
  dp_solver.cpp
  stateful_dp_solver.cpp
  parallel_stateful_oracle.cpp
  parallel_config_master_compare.cpp
)

add_executable(
  parallel_dp_guided_decision_compare
  dp_solver.cpp
  stateful_dp_solver.cpp
  parallel_stateful_oracle.cpp
  parallel_dp_guided_decision_compare.cpp
)

add_executable(
  parallel_heuristic_compare
  dp_solver.cpp
  stateful_dp_solver.cpp
  parallel_heuristic_compare.cpp
)

add_executable(
  phaseF_config_probe
  dp_solver.cpp
  phaseF_config_probe.cpp
)

add_executable(
  phaseG_config_pricer
  dp_solver.cpp
  phaseG_config_pricer.cpp
)

add_executable(
  test_lb_hierarchy
  dp_solver.cpp
  stateful_dp_solver.cpp
  test_lb_hierarchy.cpp
)

add_executable(
  rl_audit_eval
  dp_solver.cpp
  rl_audit_eval.cpp
)

# Warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(compare PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(stateful_compare PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(stateless_compare PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(parallel_pattern_compare PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(pricing_compare PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(parallel_lbbd_compare PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(parallel_frontier_runner PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(parallel_campaign_runner PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(parallel_config_master_compare PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(parallel_dp_guided_decision_compare PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(parallel_heuristic_compare PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(phaseF_config_probe PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(phaseG_config_pricer PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(test_lb_hierarchy PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(rl_audit_eval PRIVATE -Wall -Wextra -Wpedantic)
endif()

# LTO disabled: adds 2-5min link time for ~10% runtime benefit, not worth it during dev.
# To re-enable: set_property(TARGET compare PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
