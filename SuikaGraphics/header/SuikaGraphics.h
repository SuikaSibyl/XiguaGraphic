#pragma once

#include <QtWidgets/QMainWindow>
#include <QCheckBox>
#include <QProgressBar>
#include <qlabel.h>

#include "GameTimer.h"
#include "ui_SuikaGraphics.h"
#include "..\source\QDirect3D12Widget\QDirect3D12Widget.h"

class SuikaGraphics : public QMainWindow
{
    Q_OBJECT

public:
    SuikaGraphics(QWidget *parent = Q_NULLPTR);
    ~SuikaGraphics();

    void adjustWindowSize();
    void addToolbarWidgets();
    void connectSlots();

private:
    void closeEvent(QCloseEvent* event) override;

    GameTimer timer;
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
};
