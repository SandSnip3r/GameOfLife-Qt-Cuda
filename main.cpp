#include "widget.h"

#include <QApplication>

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  Widget w;
  w.show();
  w.move(0,0);
  return a.exec();
}
