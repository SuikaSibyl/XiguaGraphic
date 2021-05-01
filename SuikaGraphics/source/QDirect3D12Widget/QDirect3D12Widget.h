#pragma once

#pragma comment(lib, "D3D12.lib")
#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")

#include <stdexcept>

#include <QWidget>
#include <QTimer>
#include <QString>

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
#include <FrameResources.h>
#include <RenderItemManagment.h>
#include <InputSystem.h>
#include <Camera.h>
#include <wave.h>

#include <WorkSubmissionModule.h>
#include <MemoryManagerModule.h>
#include <SynchronizationModule.h>
#include <CudaManager.h>

using Microsoft::WRL::ComPtr;

class SuikaGraphics;

class QDirect3D12Widget : public QWidget
{
    Q_OBJECT

public:
    friend class SuikaGraphics;
    friend class MeshGeometryHelper;
    friend class RenderItem;
    friend class RenderItemManager;
    friend class TextureHelper;

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

    // ================================================================
    // ---------------------- Important Object ------------------------
    // ================================================================
    RenderItemManager RIManager;
    InputSystem InputSys;
    Camera MainCamera;

private:
    // ================================================================
    // ---------------------- Initialize Items ------------------------
    // ================================================================
    void BuildShadersAndInputLayout();
    void BuildTexture();
    void BuildMaterial();
    void BuildGeometry();
    void BuildLights();
    void BuildFrameResources();
    void BuildRootSignature();
    void BuildPSO();
    void DrawRenderItems(RenderQueue queue);
    // ------------------------------------
    void BuildBoxGeometry(); 
    void BuildMultiGeometry();
    void BuildLandGeometry();
    void BuildLakeGeometry();
    void BuildScreenCanvasGeometry();
    // ------------------------------------
    ComPtr<ID3D12RootSignature> mRootSignature = nullptr;

    // ================================================================
    // --------------------- No Important Stuffs ----------------------
    // ================================================================
    std::unique_ptr<Waves> wave;
    int vertex_num;

private:
    // ================================================================
    // ------------------ Private Important Object --------------------
    // ================================================================
    QTimer m_qTimer;        // Regularly call update
    GameTimer& m_tGameTimer; // Manage time in system

    // ================================================================
    // ----------------- Private Helper Function ----------------------
    // ================================================================
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

    // ================================================================
    // -------------------- D3D Important Module ----------------------
    // ================================================================
    std::unique_ptr<D3DModules::WorkSubmissionModule> m_WorkSubmissionModule;
    std::unique_ptr<D3DModules::MemoryManagerModule> m_MemoryManagerModule;
    std::unique_ptr<D3DModules::SynchronizationModule> m_SynchronizationModule;
    std::unique_ptr<CudaManager> m_CudaManagerModule;

    // ================================================================
    // --------------------- D3D Init Variable ------------------------
    // ================================================================
    ComPtr<ID3D12Device>    m_d3dDevice;
    ComPtr<IDXGIFactory4>   m_dxgiFactory;

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
    ID3D12CommandAllocator*         m_DirectCmdListAlloc;
    ID3D12CommandQueue*             m_CommandQueue;
    ID3D12GraphicsCommandList*      m_CommandList;
    // Resource & Swap Chain
    ComPtr<ID3D12Resource>          m_DepthStencilBuffer;
    ComPtr<IDXGISwapChain>          m_SwapChain;
    // Descriptor Heaps
    ID3D12DescriptorHeap*    m_srvHeap = nullptr;
    ComPtr<ID3D12DescriptorHeap>    m_cbvHeap = nullptr;
    // Frame resource
    FrameResource* mCurrFrameResource;

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
    void CreateRTVDSVDescriptorHeap();
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
    // init signal
    void deviceInitialized(bool success);
    // other signal
    void eventHandled();
    void widgetResized();
    // pipeline signal
    void ticked();
    void rendered(ID3D12GraphicsCommandList* cl);
    // Input signal
    void keyPressed(QKeyEvent*);
    void mouseMoved(QMouseEvent*);
    void mouseClicked(QMouseEvent*);
    void mouseReleased(QMouseEvent*);
};