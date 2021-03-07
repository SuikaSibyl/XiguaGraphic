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
#include <UploadBuffer.h>
#include <MeshGeometry.h>
#include <PipelineSetting.h>

using Microsoft::WRL::ComPtr;

class QDirect3D12Widget : public QWidget
{
    Q_OBJECT

public:
    QDirect3D12Widget(QWidget * parent);
    ~QDirect3D12Widget();

    // ================================================================
    // --------------------- LifeCycle Interface ----------------------
    // ================================================================
    void    Run();
    void    PauseFrames();
    void    ContinueFrames();
    void    Release();

    // ================================================================
    // ------------------------ Data Retrieve -------------------------
    // ================================================================
    int     GetFPS() { return m_Fps; }
    int     GetTotalTime() { return m_TotalTime; }

protected:
    // ================================================================
    // ------------------------- Life Cycle ---------------------------
    // ================================================================
    bool    Initialize();
    void    Update();
    void    Draw();

    // ================================================================
    // ----------------------- Input Callback -------------------------
    // ================================================================
    void OnMouseMove(QMouseEvent*);
    void OnMousePressed(QMouseEvent*);
    void OnMouseReleased(QMouseEvent*);

    // ================================================================
    // ----------------------- Input Callback -------------------------
    // ================================================================
    HWND const& nativeHandle() const { return m_hWnd; }
    // Render Active/Inavtive Get & Set
    bool renderActive() const { return m_bRenderActive; }
    void setRenderActive(bool active) { m_bRenderActive = active; }

private:
    // ================================================================
    // ------------------------ BOX APP only --------------------------
    // ================================================================
    D3D12_VERTEX_BUFFER_VIEW vbv;
    D3D12_INDEX_BUFFER_VIEW ibv;
    Shader* mpShader;

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

    ComPtr<ID3D12PipelineState> mPSO = nullptr;
    ComPtr<ID3D12RootSignature> mRootSignature = nullptr;
    std::unique_ptr<UploadBuffer<ObjectConstants>> mObjectCB = nullptr;
    std::unique_ptr<UploadBuffer<PassConstants>> mPassCB = nullptr;
    std::unique_ptr<MeshGeometry> mBoxGeo = nullptr;

    DirectX::XMFLOAT4X4 mWorld = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 mView = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 mProj = MathHelper::Identity4x4();

    int mLastMousePosx = 0;
    int mLastMousePosy = 0;

    float mTheta = 1.5f * DirectX::XM_PI;
    float mPhi = DirectX::XM_PIDIV4;
    float mRadius = 5.0f;

private:
    // ================================================================
    // ------------------ Private Important Object --------------------
    // ================================================================
    QTimer m_qTimer;        // Regularly call update
    GameTimer m_tGameTimer; // Manage time in system

    // ================================================================
    // ----------------- Private Helper Function ----------------------
    // ================================================================
    void FlushCmdQueue();
    void CalculateFrameState();

    ComPtr<ID3D12Resource> CreateDefaultBuffer
        (UINT64 byteSize, const void* initData, ComPtr<ID3D12Resource>& uploadBuffer);

    // ================================================================
    // ------------------ Private Helper Variable ---------------------
    // ================================================================
    HWND    m_hWnd;                 // Window Handler
    bool    m_bDeviceInitialized;
    bool    m_bStarted;
    int     m_Fps;
    int     m_TotalTime;
    bool    m_bRenderActive;        // Ture == not paused
    UINT    mCurrentBackBuffer = 0;
    int     mCurrentFence = 0;	    //Initial CPU Fence = 0

    // ================================================================
    // --------------------- D3D Init Variable ------------------------
    // ================================================================
    ComPtr<ID3D12Device>    m_d3dDevice;
    ComPtr<IDXGIFactory4>   m_dxgiFactory;
    ComPtr<ID3D12Fence>     m_fence;
    // View-Descriptor Size
    UINT m_rtvDescriptorSize;
    UINT m_dsvDescriptorSize;
    UINT m_cbv_srv_uavDescriptorSize;
    // MSAA stuff
    bool m4xMsaaState = false;  // 4X MSAA enabled
    UINT m4xMsaaQuality = 0;    // quality level of 4X MSAA
    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS msaaQualityLevels;
    DXGI_FORMAT m_BackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    DXGI_FORMAT m_DepthStencilFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
    // Command Objects
    ComPtr<ID3D12CommandAllocator>      m_DirectCmdListAlloc;
    ComPtr<ID3D12CommandQueue>          m_CommandQueue;
    ComPtr<ID3D12GraphicsCommandList>   m_CommandList;
    // Resource & Swap Chain
    ComPtr<ID3D12Resource>          m_DepthStencilBuffer;
    ComPtr<ID3D12Resource>          m_SwapChainBuffer[2];
    ComPtr<IDXGISwapChain>          m_SwapChain;
    // Descriptor Heaps
    ComPtr<ID3D12DescriptorHeap>    m_rtvHeap = nullptr;
    ComPtr<ID3D12DescriptorHeap>    m_dsvHeap = nullptr;
    ComPtr<ID3D12DescriptorHeap>    m_cbvHeap = nullptr;
    // Rect & ViewPort
    D3D12_RECT scissorRect;
    D3D12_VIEWPORT viewPort;

    // ================================================================
    // --------------------- D3D Initilization ------------------------
    // ================================================================
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

    // ================================================================
    // -------------------------- Qt Events ---------------------------
    // ================================================================
    bool            event(QEvent* event) override;
    void            showEvent(QShowEvent* event) override;
    QPaintEngine*   paintEngine() const override;
    void            paintEvent(QPaintEvent* event) override;
    void            resizeEvent(QResizeEvent* event) override;
    void            wheelEvent(QWheelEvent* event) override;

#if QT_VERSION >= 0x050000
    bool            nativeEvent(const QByteArray& eventType, void* message, long* result) override;
#else
    bool            winEvent(MSG* message, long* result) override;
#endif
    LRESULT WINAPI WndProc(MSG* pMsg);

    // ================================================================
    // -------------------------- Qt Slots ----------------------------
    // ================================================================
private slots:
    void onFrame();
    void onResize();

    // ================================================================
    // ------------------------- Qt Signals ---------------------------
    // ================================================================
signals:
    void deviceInitialized(bool success);

    void eventHandled();
    void widgetResized();

    void ticked();
    void rendered(ID3D12GraphicsCommandList* cl);

    void keyPressed(QKeyEvent*);
    void mouseMoved(QMouseEvent*);
    void mouseClicked(QMouseEvent*);
    void mouseReleased(QMouseEvent*);
};