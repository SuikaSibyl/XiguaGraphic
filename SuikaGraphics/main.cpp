#include "SuikaGraphics.h"
#include <QtWidgets/QApplication>

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    SuikaGraphics w;
    w.show();
    return a.exec();
}
