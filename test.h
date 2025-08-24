#ifndef TEST_H_
#define TEST_H_

#include <cuda_runtime.h>

__global__
void updateGrid(bool *currGrid, bool *nextGrid, int kWidth, int kHeight);

void cudaNextStep(bool *currGrid, bool *nextGrid, int kWidth, int kHeight);

#endif // TEST_H_