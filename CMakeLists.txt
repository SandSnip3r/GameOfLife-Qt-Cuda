cmake_minimum_required(VERSION 3.28)

project(GameOfLife
  VERSION 0.1
  LANGUAGES CXX CUDA
)

# ---- Standards ---------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the local GPU by default; override with -DCMAKE_CUDA_ARCHITECTURES="75;86;89;90"
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")

# ---- Qt (GUI app only) -------------------------------------------------------
find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(CUDAToolkit REQUIRED)
qt_standard_project_setup()

# ---- CUDA kernels library ----------------------------------------------------
# Put all device kernels used by both binaries here.
# If you later split them out, just add more .cu files to this target.
add_library(cuda_kernels STATIC
  test.cu
)
set_target_properties(cuda_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(cuda_kernels PUBLIC CUDA::cudart)
target_include_directories(cuda_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}  # so headers next to .cu are visible
)

# Optional: CUDA-specific flags
target_compile_options(cuda_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
)

# ---- GUI executable (Qt Widgets) --------------------------------------------
set(APP_SOURCES
  main.cpp
  widget.cpp
  widget.h
  widget.ui
  # NOTE: kernels are no longer listed here; they live in cuda_kernels
)

qt_add_executable(game_of_life
  MANUAL_FINALIZATION
  ${APP_SOURCES}
)

# C++ warnings (apply only to C++ files, not CUDA host compiles)
if (MSVC)
  target_compile_options(game_of_life PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /utf-8>
  )
else()
  target_compile_options(game_of_life PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  )
endif()

target_link_libraries(game_of_life PRIVATE
  Qt6::Widgets
  cuda_kernels
)

qt_finalize_executable(game_of_life)

# ---- Headless benchmark executable (no Qt) -----------------------------------
# A minimal, UI-less binary you can profile with nsys/ncu.
# Source file you mentioned: benchmark.cu
add_executable(benchmark
  benchmark.cu
)

# If benchmark.cu calls into kernels defined in test.cu, it links the same lib:
target_link_libraries(benchmark PRIVATE
  cuda_kernels
  CUDA::cudart
)

# Optional: apply only CUDA flags to the benchmark TU (keeps host warnings quiet)
target_compile_options(benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
)

# Nice: group in IDEs
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
  ${APP_SOURCES} benchmark.cu test.cu
)
