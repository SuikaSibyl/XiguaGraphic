#pragma once

#include <QtWidgets/QMainWindow>
#include <QCheckBox>
#include <QProgressBar>
#include <qlabel.h>
#include <QTextBrowser>
#include <QString>
#include <string>

#include "ui_SuikaGraphics.h"
#include <QDirect3D12Widget.h>

class SuikaGraphics : public QMainWindow
{
    Q_OBJECT

public:
    void AppendDebugInfo(QString info);

public:
    SuikaGraphics(QWidget *parent = Q_NULLPTR);
    ~SuikaGraphics();

    void adjustWindowSize();
    void addToolbarWidgets();
    void connectSlots();

private:
    void closeEvent(QCloseEvent* event) override;

    // UI Components
    QProgressBar* progressBar1;
    QLabel* fpsShower;
    QLabel* timeShower;


public slots:
    void init(bool success);
    void tick();
    void render(ID3D12GraphicsCommandList* cl);

private:
    Ui::SuikaGraphicsClass* ui;

    QDirect3D12Widget*  m_pScene;
    QSize               m_WindowSize;
    QCheckBox*          m_pCbxDoFrames;
    QTextBrowser*       m_pDebugTxt;
};
