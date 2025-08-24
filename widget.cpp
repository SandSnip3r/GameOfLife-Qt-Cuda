#include "test.h"
#include "widget.h"
#include "./ui_widget.h"

#include <cuda_runtime.h>

#include <QPainter>

#include <cstring>
#include <chrono>
#include <iostream>

int Widget::toIndex(int row, int col) {
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

Widget::Widget(QWidget *parent) : QWidget(parent), ui(new Ui::Widget) {
  ui->setupUi(this);

  // Window properties
  setMinimumSize(kWidth,kHeight);

  // GPU grid memory
  cudaMalloc(&dArray1_, kWidth*kHeight*sizeof(bool));
  cudaMalloc(&dArray2_, kWidth*kHeight*sizeof(bool));

  // CPU grid memory
  array1_ = new bool[kWidth*kHeight];

  // Create a QImage to be used with the QPainter for quick drawing
  img_ = new QImage(kWidth, kHeight, QImage::Format_ARGB32);

  initializeGame();
}

void Widget::initializeGame() {
  // Fill array1_ with the initial conditions

  // Set all to false
  std::memset(array1_, 0, kWidth*kHeight);

  // Set some to true to start something interesting
  // 1,2,3,4,5,6,7,8,9 // Cool flying diagonals
  for (auto col : { kWidth/4, kWidth/2, 3*kWidth/4 }) {
    for (int row=0; row<kHeight; ++row) {
      array1_[toIndex(row,col)] = true;
    }
  }
  // 1,2,3,4,5,6,7,8,9 // Cool flying diagonals
  for (auto row : { kHeight/4, kHeight/2, 3*kHeight/4 }) {
    for (int col=0; col<kWidth; ++col) {
      array1_[toIndex(row,col)] = true;
    }
  }

  // Copy grid to GPU
  cudaMemcpy(dArray1_, array1_, kWidth*kHeight*sizeof(bool), cudaMemcpyHostToDevice);

  fillImage();
  update();
}

void Widget::nextIteration() {
  // Copy grid from GPU
  cudaMemcpy(array1_, dArray1_, kWidth*kHeight*sizeof(bool), cudaMemcpyDeviceToHost);

  // Update image data based on array of bools
  fillImage();

  // Update grid
  cudaNextStep(dArray1_, dArray2_, kWidth, kHeight);

  // Roll buffers
  std::swap(dArray1_, dArray2_);

  update();
}

void Widget::fillImage() {
  // Always set image based on array1_
  QRgb *data = reinterpret_cast<QRgb*>(img_->bits());
  for (int row=0; row<kHeight; ++row) {
    // Create a simple gradient from top to bottom
    const int blueValue = 255 * static_cast<double>(row)/(kHeight+1);
    for (int col=0; col<kWidth; ++col) {
      const auto index = toIndex(row,col);
      if (array1_[index]) {
        // Create a simple gradient from left to right
        const int redValue = 255 * static_cast<double>(col)/(kWidth+1);
        // Cell is alive, color
        data[index] = qRgb(redValue,255,255-blueValue);
      } else {
        // Cell is dead, set color as black
        data[index] = qRgb(0,0,0);
      }
    }
  }
}

void Widget::paintEvent(QPaintEvent *event) {
  (void)event;
  QPainter painter(this);
  QRectF source(0,0,img_->width(), img_->height());
  QRectF target(0,0,img_->width(), img_->height());
  painter.drawImage(target, *img_, source);
  if (go_) {
    nextIteration();
    if (kStepFrameByFrame) {
      go_ = false;
    }
  }
}

void Widget::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    go_ = true;
  }
  nextIteration();
}

Widget::~Widget() {
  delete ui;
  delete[] array1_;
  cudaFree(dArray1_);
  cudaFree(dArray2_);
}
