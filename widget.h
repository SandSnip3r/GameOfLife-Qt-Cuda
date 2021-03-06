#ifndef WIDGET_H
#define WIDGET_H

#include <QImage>
#include <QWidget>

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

private:
  static constexpr int kWidth=2048, kHeight=1024;
  Ui::Widget *ui;
  bool *array1_;
  bool *dArray1_, *dArray2_;
  QImage *img_;

  static int toIndex(int row, int col);
  void fillImage();
  void initializeGame();
  void nextIteration();
};
#endif // WIDGET_H
