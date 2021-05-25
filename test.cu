#include "test.h"

#include <cuda_runtime.h>

__device__
int toIndex(int row, int col, int kWidth, int kHeight) {
  if (row < 0) {
    row += kHeight;
  } else if (row >= kHeight) {
    row -= kHeight;
  }
  if (col < 0) {
    col += kWidth;
  } else if (col >= kWidth) {
    col -= kWidth;
  }
  return row*kWidth + col;
}

__global__
void updateGrid(bool *currGrid, bool *nextGrid, int kWidth, int kHeight) {
  constexpr bool wrap{true};
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i=idx; i<kHeight*kWidth; i+=stride) {
    const int row = i / kWidth;
    const int col = i % kWidth;
    
    int livingNeighbors = 0;
    bool ne=true, se=true, sw=true, nw=true;
    if (row != 0 || wrap) {
      // Check up
      if (currGrid[toIndex(row-1, col, kWidth, kHeight)]) {
        ++livingNeighbors;
      }
    } else {
      ne = false;
      nw = false;
    }
    if (row < kHeight || wrap) {
      // Check down
      if (currGrid[toIndex(row+1, col, kWidth, kHeight)]) {
        ++livingNeighbors;
      }
    } else {
      se = false;
      sw = false;
    }
    if (col != 0 || wrap) {
      // Check left
      if (currGrid[toIndex(row, col-1, kWidth, kHeight)]) {
        ++livingNeighbors;
      }
    } else {
      sw = false;
      nw = false;
    }
    if (col < kWidth || wrap) {
      // Check right
      if (currGrid[toIndex(row, col+1, kWidth, kHeight)]) {
        ++livingNeighbors;
      }
    } else {
      ne = false;
      se = false;
    }
    if (ne) {
      // Check northeast
      if (currGrid[toIndex(row-1, col+1, kWidth, kHeight)]) {
        ++livingNeighbors;
      }
    }
    if (se) {
      // Check southeast
      if (currGrid[toIndex(row+1, col+1, kWidth, kHeight)]) {
        ++livingNeighbors;
      }
    }
    if (sw) {
      // Check southwest
      if (currGrid[toIndex(row+1, col-1, kWidth, kHeight)]) {
        ++livingNeighbors;
      }
    }
    if (nw) {
      // Check northwest
      if (currGrid[toIndex(row-1, col-1, kWidth, kHeight)]) {
        ++livingNeighbors;
      }
    }
    
    // Now, do something based on neighbor count
    if (currGrid[toIndex(row,col, kWidth, kHeight)]) {
      // Living already
      if (livingNeighbors < 2) {
        // Any live cell with fewer than two live neighbours dies, as if by underpopulation.
        nextGrid[toIndex(row,col, kWidth, kHeight)] = false;
      } else if (livingNeighbors < 4) {
        // Any live cell with two or three live neighbours lives on to the next generation.
        nextGrid[toIndex(row,col, kWidth, kHeight)] = true;
      } else {
        // Any live cell with more than three live neighbours dies, as if by overpopulation.
        nextGrid[toIndex(row,col, kWidth, kHeight)] = false;
      }
    } else {
      // Dead
      if (livingNeighbors == 3) {
        // Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        nextGrid[toIndex(row,col, kWidth, kHeight)] = true;
      } else {
        // Stay dead
        nextGrid[toIndex(row,col, kWidth, kHeight)] = false;
      }
    }
  }
}

void cudaNextStep(bool *currGrid, bool *nextGrid, int kWidth, int kHeight) {
  updateGrid<<<256,256>>>(currGrid, nextGrid, kWidth, kHeight);
}
