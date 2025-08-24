#include "test.h"

#include <cuda.h>

#include <iostream>
#include <vector>

using namespace std;

int main() {
  constexpr int kWidth=2048, kHeight=1024;

  std::vector<char> hostGrid(kWidth*kHeight, 0);
  // Initialize one single row
  constexpr int kRow = kHeight/2;
  for (int col=0; col<kWidth; ++col) {
    hostGrid[kRow*kWidth + col] = 1;
  }

  bool *deviceArray1, *deviceArray2;
  cudaMalloc(&deviceArray1, sizeof(bool)*kWidth*kHeight);
  cudaMalloc(&deviceArray2, sizeof(bool)*kWidth*kHeight);

  // Populate the first array
  cudaMemcpy(deviceArray1, hostGrid.data(), sizeof(bool)*kWidth*kHeight, cudaMemcpyHostToDevice);

  // Run for some iterations
  int block = 512;
  int grid = ((kWidth*kHeight)+1) / block;
  // dim3 block(32,32);
  // dim3 grid((kWidth+1)/block.x, (kHeight+1)/block.y);
  for (int i=0; i<1; ++i) {
    updateGrid<<<grid, block>>>(deviceArray1, deviceArray2, kWidth, kHeight);
    std::swap(deviceArray1, deviceArray2);
  }

  // Copy result back
  cudaMemcpy(hostGrid.data(), deviceArray1, sizeof(bool)*kWidth*kHeight, cudaMemcpyDeviceToHost);

  // Do a quick sanity check to make sure we computed the right thing.
  int count=0;
  for (char c : hostGrid) {
    if (c == 1) {
      ++count;
    } else if (c != 0) {
      cout << "Invalid value: " << static_cast<int>(c) << endl;
    }
  }
  cout << "Count: " << count << endl;
  return 0;
}