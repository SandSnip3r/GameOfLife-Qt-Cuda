#include "test.h"
#include "widget.h"
#include "./ui_widget.h"

#include <cuda_runtime.h>

#include <QPainter>

#include <cstring>
#include <chrono>
#include <iostream>

namespace {
int toIndex(int row, int col, int width, int height) {
  if (row < 0) {
    row += height;
  } else if (row >= height) {
    row -= height;
  }
  if (col < 0) {
    col += width;
  } else if (col >= width) {
    col -= width;
  }
  return row*width + col;
}
} // anonymous namespace

Widget::Widget(QWidget *parent) : QWidget(parent), ui(new Ui::Widget) {
  ui->setupUi(this);

  // Window properties
  setMinimumSize(kGridWidth*kRenderPixelSize,kGridHeight*kRenderPixelSize);

  // GPU grid memory
  cudaMalloc(&dArray1_, kGridWidth*kGridHeight*sizeof(bool));
  cudaMalloc(&dArray2_, kGridWidth*kGridHeight*sizeof(bool));

  // CPU grid memory
  array1_ = new bool[kGridWidth*kGridHeight];

  // Create a QImage to be used with the QPainter for quick drawing
  img_ = new QImage(kGridWidth*kRenderPixelSize,
                    kGridHeight*kRenderPixelSize,
                    QImage::Format_ARGB32);

  initializeGame();
}

void Widget::initializeGame() {
  // Fill array1_ with the initial conditions
  // Set all to false
  std::memset(array1_, 0, kGridWidth*kGridHeight);

  // Set some to true to start something interesting
  // 1,2,3,4,5,6,7,8,9 // Cool flying diagonals
  for (auto col : { kGridWidth/4, kGridWidth/2, 3*kGridWidth/4 }) {
    for (int row=0; row<kGridHeight; ++row) {
      array1_[toIndex(row, col, kGridWidth, kGridHeight)] = true;
    }
  }
  // 1,2,3,4,5,6,7,8,9 // Cool flying diagonals
  for (auto row : { kGridHeight/4, kGridHeight/2, 3*kGridHeight/4 }) {
    for (int col=0; col<kGridWidth; ++col) {
      array1_[toIndex(row, col, kGridWidth, kGridHeight)] = true;
    }
  }

  // Copy grid to GPU
  cudaMemcpy(dArray1_, array1_, kGridWidth*kGridHeight*sizeof(bool), cudaMemcpyHostToDevice);

  fillImage();
  update();
}

void Widget::nextIteration() {
  // Copy grid from GPU
  cudaMemcpy(array1_, dArray1_, kGridWidth*kGridHeight*sizeof(bool), cudaMemcpyDeviceToHost);

  // Update image data based on array of bools
  fillImage();

  // Update grid
  cudaNextStep(dArray1_, dArray2_, kGridWidth, kGridHeight);

  // Roll buffers
  std::swap(dArray1_, dArray2_);

  if (kStepFrameByFrame) {
    go_ = false;
  }

  update();
}

void Widget::fillImage() {
  // Always set image based on array1_
  QRgb *data = reinterpret_cast<QRgb*>(img_->bits());
  for (int gridRow=0; gridRow<kGridHeight; ++gridRow) {
    // Create a simple gradient from top to bottom
    const int blueValue = 255 * static_cast<double>(gridRow)/(kGridHeight+1);
    for (int gridCol=0; gridCol<kGridWidth; ++gridCol) {
      const int redValue = 255 * static_cast<double>(gridCol)/(kGridWidth+1);
      const auto gridIndex = toIndex(gridRow, gridCol, kGridWidth, kGridHeight);
      const QRgb pixelColor = array1_[gridIndex] ? qRgb(redValue,255,255-blueValue) : qRgb(0,0,0);
      for (int pixelRow=0; pixelRow<kRenderPixelSize; ++pixelRow) {
        for (int pixelCol=0; pixelCol<kRenderPixelSize; ++pixelCol) {
          const int imgRow = gridRow*kRenderPixelSize + pixelRow;
          const int imgCol = gridCol*kRenderPixelSize + pixelCol;
          const auto imgIndex = toIndex(imgRow, imgCol, img_->width(), img_->height());
          data[imgIndex] = pixelColor;
        }
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
