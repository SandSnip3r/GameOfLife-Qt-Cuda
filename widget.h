#ifndef WIDGET_H
#define WIDGET_H

#include <QImage>
#include <QWidget>
#include <QMouseEvent>

#include <vector>

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget {
  Q_OBJECT

public:
  Widget(QWidget *parent = nullptr);
  ~Widget();

protected:
  virtual void paintEvent(QPaintEvent *event) override;
  virtual void mousePressEvent(QMouseEvent *event) override;

private:
  static constexpr int kGridWidth=2049, kGridHeight=1025;
  static constexpr int kRenderPixelSize=1;
  static constexpr bool kStepFrameByFrame{false};
  Ui::Widget *ui;

  // CPU grid memory
  std::vector<char> array1_;

  // GPU grid memory
  bool *dArray1_, *dArray2_;

  QImage *img_;
  bool go_{true};

  void fillImage();
  void initializeGame();
  void nextIteration();
};
#endif // WIDGET_H
