#include "SuikaGraphics.h"

#include <QStyle>
#include <QDebug>
#include <QTime>
#include <QScreen>
#include <QMessageBox>
#include <QCloseEvent>
#include <QDesktopWidget>

#include <iostream>

SuikaGraphics::SuikaGraphics(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::SuikaGraphicsClass)
    , m_WindowSize(QSize(1280, 800))
    , m_pCbxDoFrames(new QCheckBox(this))
{
    ui->setupUi(this);
    m_pScene = ui->view;
    setCentralWidget(m_pScene);


    progressBar1 = new QProgressBar;
    progressBar1->setMaximumWidth(200);
    progressBar1->setMinimum(0);
    progressBar1->setMaximum(100);
    progressBar1->setValue(100);
    ui->statusBar->addWidget(progressBar1);

    fpsShower = new QLabel;
    fpsShower->setText(QString("fps: ") + QString::number(60));
    fpsShower->setMinimumWidth(100);
    fpsShower->setMaximumWidth(200);
    ui->statusBar->addWidget(fpsShower);

    timeShower = new QLabel;
    timeShower->setText(QString("run time: ") + QString::number(0));
    timeShower->setMinimumWidth(100);
    timeShower->setMaximumWidth(200);
    ui->statusBar->addWidget(timeShower);

    adjustWindowSize();
    addToolbarWidgets();
    connectSlots();

    timer.Start();
    timer.Reset();
}

SuikaGraphics::~SuikaGraphics() = default;

void SuikaGraphics::adjustWindowSize()
{
    resize(m_WindowSize.width(), m_WindowSize.height());
    setGeometry(QStyle::alignedRect(Qt::LeftToRight, Qt::AlignCenter, size(),
        qApp->screens().first()->availableGeometry()));
}

void SuikaGraphics::addToolbarWidgets()
{
    // Add CheckBox to tool-bar to stop/continue frames execution.
    m_pCbxDoFrames->setText("Do Frames");
    m_pCbxDoFrames->setChecked(true);
    connect(m_pCbxDoFrames, &QCheckBox::stateChanged, [&] {
        if (m_pCbxDoFrames->isChecked())
            m_pScene->continueFrames();
        else
            m_pScene->pauseFrames();
        });
    ui->mainToolBar->addWidget(m_pCbxDoFrames);
}

void SuikaGraphics::connectSlots()
{
    connect(m_pScene, &QDirect3D12Widget::deviceInitialized, this, &SuikaGraphics::init);
    connect(m_pScene, &QDirect3D12Widget::ticked, this, &SuikaGraphics::tick);
    connect(m_pScene, &QDirect3D12Widget::rendered, this, &SuikaGraphics::render);

    // NOTE: Additionally, you can listen to some basic IO events.
    // connect(m_pScene, &QDirect3D12Widget::keyPressed, this, &MainWindow::onKeyPressed);
    // connect(m_pScene, &QDirect3D12Widget::mouseMoved, this, &MainWindow::onMouseMoved);
    // connect(m_pScene, &QDirect3D12Widget::mouseClicked, this, &MainWindow::onMouseClicked);
    // connect(m_pScene, &QDirect3D12Widget::mouseReleased, this,
    // &MainWindow::onMouseReleased);
}

void SuikaGraphics::init(bool success)
{
    if (!success)
    {
        QMessageBox::critical(this, "ERROR", "Direct3D widget initialization failed.",
            QMessageBox::Ok);
        return;
    }

    // TODO: Add here your extra initialization here.
    // ...

    // Start processing frames with a short delay in case things are still initializing/loading
    // in the background.
    QTimer::singleShot(500, this, [&] { m_pScene->run(); });
    disconnect(m_pScene, &QDirect3D12Widget::deviceInitialized, this, &SuikaGraphics::init);
}

static int frameCnt = 0;
static float timeElapsed = 0.0f;

void SuikaGraphics::tick()
{
    // TODO: Update the scene here.
    // m_pMesh->Tick();

    timer.Tick();
    frameCnt++;

    // Compute averages over one second period.
            //timer.TotalTime() - timeElapsed >= 1.0f代表度过了一秒的时间
            //在一秒末统计这一秒刷过的帧数
    if ((timer.TotalTime() - timeElapsed) >= 1.0f)
    {
        float fps = (float)frameCnt; // fps = frameCnt / 1
        float mspf = 1000.0f / fps;

        fpsShower->setText(QString("fps: ") + QString::number(fps));
        timeShower->setText(QString("run time: ") + QString::number(int(timer.TotalTime())));

        // Reset for next average.
        frameCnt = 0;
        timeElapsed += 1.0f;
    }
}

void SuikaGraphics::render(ID3D12GraphicsCommandList* cl)
{
    // TODO: Use the command list pointer to queue items to be rendered.
}

void SuikaGraphics::closeEvent(QCloseEvent * event)
{
    event->ignore();
    m_pScene->release();
    QTime dieTime = QTime::currentTime().addMSecs(500);
    while (QTime::currentTime() < dieTime)
        QCoreApplication::processEvents(QEventLoop::AllEvents, 100);

    event->accept();
}
