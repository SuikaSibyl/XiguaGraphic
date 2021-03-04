#pragma once

#pragma comment(lib, "D3D12.lib")
#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")

#include <stdexcept>

#include <QWidget>
#include <QTimer>

#include <wrl.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <D3Dcompiler.h>

#include <Utility.h>
#include <GameTimer.h>

#include <Shader.h>

using Microsoft::WRL::ComPtr;

class QDirect3D12Widget : public QWidget
{
    Q_OBJECT

public:
    QDirect3D12Widget(QWidget * parent);
    ~QDirect3D12Widget();

    void release();
    void run();
    void pauseFrames();
    void continueFrames();

public:
    int m_Fps;
    int m_TotalTime;

private:
    // Lifespan
    bool init();

    void tick();
    bool InitDirect3D();
#pragma region InitializeSubFunctions
    void CreateDevice();
    void CreateFence();
    void GetDescriptorSize();
    void SetMSAA();
    void CreateCommandObjects();
    void CreateSwapChain();
    void CreateDescriptorHeap();
    void CreateRTV();
    void CreateDSV();
    void CreateViewPortAndScissorRect();
#pragma endregion

    void FlushCmdQueue();
    void CalculateFrameState();
    void Draw();
    void CreateBuffer();
    ComPtr<ID3D12Resource> CreateDefaultBuffer
        (UINT64 byteSize, const void* initData, ComPtr<ID3D12Resource>& uploadBuffer);

    ComPtr<ID3D12PipelineState> PSO;
    ComPtr<ID3D12DescriptorHeap> mCbvHeap = nullptr;
    D3D12_VERTEX_BUFFER_VIEW vbv;
    D3D12_INDEX_BUFFER_VIEW ibv;
    Shader* mpShader;

protected:
    int mCurrentFence = 0;	//初始CPU上的围栏点为0

    GameTimer timer;

    /// <summary>
    /// 声明指针接口和变量
    /// </summary>
    // ====================================================
    D3D12_VIEWPORT viewPort;
    D3D12_RECT scissorRect;

    UINT mCurrentBackBuffer = 0;

    UINT rtvDescriptorSize;
    UINT dsvDescriptorSize;
    UINT cbv_srv_uavDescriptorSize;

    void DrawBox(const GameTimer& gt);

    void BuildDescriptorHeaps();
    void BuildConstantBuffers();
    void BuildRootSignature();
    void BuildShadersAndInputLayout();
    void BuildBoxGeometry();
    void BuildPSO();
    //Default buffer
    ComPtr<ID3D12Resource> VertexBufferGPU = nullptr;
    ComPtr<ID3D12Resource> IndexBufferGPU = nullptr;
    //Upload buffer
    ComPtr<ID3D12Resource> VertexBufferUploader = nullptr;
    ComPtr<ID3D12Resource> IndexBufferUploader = nullptr;

    ComPtr<ID3D12Device>    md3dDevice;
    ComPtr<IDXGIFactory4>   dxgiFactory;
    ComPtr<ID3D12Fence>     fence;

    ComPtr<ID3D12CommandAllocator>      mDirectCmdListAlloc;
    ComPtr<ID3D12CommandQueue>          mCommandQueue;
    ComPtr<ID3D12GraphicsCommandList>   mCommandList;

    ComPtr<ID3D12Resource>          depthStencilBuffer;
    ComPtr<ID3D12Resource>          swapChainBuffer[2];
    ComPtr<IDXGISwapChain>          m_SwapChain;
    ComPtr<ID3D12DescriptorHeap>    rtvHeap;
    ComPtr<ID3D12DescriptorHeap>    dsvHeap;
    // ====================================================


    // Qt Events
private:
    bool           event(QEvent * event) override;
    void           showEvent(QShowEvent * event) override;
    QPaintEngine * paintEngine() const override;
    void           paintEvent(QPaintEvent * event) override;
    void           resizeEvent(QResizeEvent * event) override;
    void           wheelEvent(QWheelEvent * event) override;

    LRESULT WINAPI WndProc(MSG * pMsg);

#if QT_VERSION >= 0x050000
    bool nativeEvent(const QByteArray & eventType, void * message, long * result) override;
#else
    bool winEvent(MSG * message, long * result) override;
#endif

signals:
    void deviceInitialized(bool success);

    void eventHandled();
    void widgetResized();

    void ticked();
    void rendered(ID3D12GraphicsCommandList * cl);

    void keyPressed(QKeyEvent *);
    void mouseMoved(QMouseEvent *);
    void mouseClicked(QMouseEvent *);
    void mouseReleased(QMouseEvent *);

private slots:
    void onFrame();
    void onReset();

    // Getters / Setters
public:
    HWND const & nativeHandle() const { return m_hWnd; }

    bool renderActive() const { return m_bRenderActive; }
    void setRenderActive(bool active) { m_bRenderActive = active; }

    D3DCOLORVALUE * BackColor() { return &m_BackColor; }

protected:

    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS msaaQualityLevels;

    // Widget objects.
    QTimer m_qTimer;

    HWND m_hWnd;
    bool m_bDeviceInitialized;
    bool m_bRenderActive;
    bool m_bStarted;

    D3DCOLORVALUE m_BackColor;
};