# Game of Life

This is a simulation of Conway's Game of Life. The visualization framework is Qt. This application leverages Nvidia GPUs to quickly compute the next frame.

<p align="center">
  <img src="./examples/game-of-life.png" width="1024" title="Example">
</p>

# Building

At the end of CMakeLists.txt, make sure to modify the `CUDA_ARCHITECTURES` to match that of the GPU of your machine. Also, there is a bug in the Qt CMake system before version 5.14.1/5.15.0 Alpha that will cause an issue. Be mindful to have a new version.
