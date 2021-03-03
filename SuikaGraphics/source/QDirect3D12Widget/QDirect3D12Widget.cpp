#include "QDirect3D12Widget.h"
#include <DirectXColors.h>

#include <QDebug>
#include <QEvent>
#include <QWheelEvent>

using Microsoft::WRL::ComPtr;

constexpr int FPS_LIMIT    = 120.0f;
constexpr int MS_PER_FRAME = (int)((1.0f / FPS_LIMIT) * 1000.0f);

QDirect3D12Widget::QDirect3D12Widget(QWidget * parent)
    : QWidget(parent)
    , m_iCurrFrameIndex(0)
    , m_pDevice(Q_NULLPTR)
    , m_pFactory(Q_NULLPTR)
    , m_pSwapChain(Q_NULLPTR)
    , m_pCommandQueue(Q_NULLPTR)
    , m_pCommandAllocators{}
    , m_pCommandList(Q_NULLPTR)
    , m_pRTVDescHeap(Q_NULLPTR)
    , m_iRTVDescSize(0)
    , m_pRTVResources{}
    , m_RTVDescriptors{}
    , m_pSrvDescHeap(Q_NULLPTR)
    , m_hSwapChainEvent(Q_NULLPTR)
    , m_hFenceEvent(Q_NULLPTR)
    , m_pFence(Q_NULLPTR)
    , m_iFenceValues{}
    , m_hWnd(reinterpret_cast<HWND>(winId()))
    , m_bDeviceInitialized(false)
    , m_bRenderActive(false)
    , m_bStarted(false)
    , m_BackColor{0.0f, 0.135f, 0.481f, 1.0f}
{
    qDebug() << "[QDirect3D12Widget::QDirect3D12Widget] - Widget Handle: " << m_hWnd;

    QPalette pal = palette();
    pal.setColor(QPalette::Window, Qt::black);
    setAutoFillBackground(true);
    setPalette(pal);

    setFocusPolicy(Qt::StrongFocus);
    setAttribute(Qt::WA_NativeWindow);

    // Setting these attributes to our widget and returning null on paintEngine event
    // tells Qt that we'll handle all drawing and updating the widget ourselves.
    setAttribute(Qt::WA_PaintOnScreen);
    setAttribute(Qt::WA_NoSystemBackground);
}

QDirect3D12Widget::~QDirect3D12Widget() {}

void QDirect3D12Widget::release()
{
    m_bDeviceInitialized = false;
    disconnect(&m_qTimer, &QTimer::timeout, this, &QDirect3D12Widget::onFrame);
    m_qTimer.stop();

    //waitForGpu();
}

void QDirect3D12Widget::run()
{
    m_qTimer.start(MS_PER_FRAME);
    m_bRenderActive = m_bStarted = true;
}

void QDirect3D12Widget::pauseFrames()
{
    if (!m_qTimer.isActive() || !m_bStarted) return;

    disconnect(&m_qTimer, &QTimer::timeout, this, &QDirect3D12Widget::onFrame);
    m_qTimer.stop();
    m_bRenderActive = false;
}

void QDirect3D12Widget::continueFrames()
{
    if (m_qTimer.isActive() || !m_bStarted) return;

    connect(&m_qTimer, &QTimer::timeout, this, &QDirect3D12Widget::onFrame);
    m_qTimer.start(MS_PER_FRAME);
    m_bRenderActive = true;
}

void QDirect3D12Widget::showEvent(QShowEvent * event)
{
    if (!m_bDeviceInitialized)
    {
        m_bDeviceInitialized = init();
        emit deviceInitialized(m_bDeviceInitialized);
    }

    QWidget::showEvent(event);
}

bool QDirect3D12Widget::init()
{
    InitDirect3D();

    connect(&m_qTimer, &QTimer::timeout, this, &QDirect3D12Widget::onFrame);

    timer.Reset();

    return true;
}

/// <summary>
/// 
/// </summary>
void QDirect3D12Widget::onFrame()
{
    // Send ticked signal
    if (m_bRenderActive) tick();

    timer.Tick();

    CalculateFrameState();
    Draw();
}

void QDirect3D12Widget::FlushCmdQueue()
{
    mCurrentFence++;	//CPU传完命令并关闭后，将当前围栏值+1
    mCommandQueue->Signal(fence.Get(), mCurrentFence);	//当GPU处理完CPU传入的命令后，将fence接口中的围栏值+1，即fence->GetCompletedValue()+1
    if (fence->GetCompletedValue() < mCurrentFence)	//如果小于，说明GPU没有处理完所有命令
    {
        HANDLE eventHandle = CreateEvent(nullptr, false, false, L"FenceSetDone");	//创建事件
        fence->SetEventOnCompletion(mCurrentFence, eventHandle);//当围栏达到mCurrentFence值（即执行到Signal（）指令修改了围栏值）时触发的eventHandle事件
        WaitForSingleObject(eventHandle, INFINITE);//等待GPU命中围栏，激发事件（阻塞当前线程直到事件触发，注意此Enent需先设置再等待，
                               //如果没有Set就Wait，就死锁了，Set永远不会调用，所以也就没线程可以唤醒这个线程）
        CloseHandle(eventHandle);
    }
}
/// <summary>
/// 
/// </summary>
void QDirect3D12Widget::CalculateFrameState()
{
    static int frameCnt = 0;	//总帧数
    static float timeElapsed = 0.0f;	//流逝的时间
    frameCnt++;	//每帧++，经过一秒后其即为FPS值
    //调试模块
    /*std::wstring text = std::to_wstring(gt.TotalTime());
    std::wstring windowText = text;
    SetWindowText(mhMainWnd, windowText.c_str());*/
    //判断模块
    if (timer.TotalTime() - timeElapsed >= 1.0f)	//一旦>=0，说明刚好过一秒
    {
        float fps = (float)frameCnt;//每秒多少帧
        float mspf = 1000.0f / fps;	//每帧多少毫秒

        std::wstring fpsStr = std::to_wstring(fps);//转为宽字符
        std::wstring mspfStr = std::to_wstring(mspf);
        //将帧数据显示在窗口上
        std::wstring windowText = L"D3D12Init    fps:" + fpsStr + L"    " + L"mspf" + mspfStr;
        SetWindowText((HWND)winId(), windowText.c_str());

        //为计算下一组帧数值而重置
        frameCnt = 0;
        timeElapsed += 1.0f;
    }
}
void QDirect3D12Widget::Draw()
{
    //首先重置命令分配器cmdAllocator和命令列表cmdList，目的是重置命令和列表，复用相关内存。
    ThrowIfFailed(mDirectCmdListAlloc->Reset());//重复使用记录命令的相关内存
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));//复用命令列表及其内存

    //接着我们将后台缓冲资源从呈现状态转换到渲染目标状态（即准备接收图像渲染）。
    UINT& ref_mCurrentBackBuffer = mCurrentBackBuffer;
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(swapChainBuffer[ref_mCurrentBackBuffer].Get(),//转换资源为后台缓冲区资源
        D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));//从呈现到渲染目标转换

    //接下来设置视口和裁剪矩形。
    mCommandList->RSSetViewports(1, &viewPort);
    mCommandList->RSSetScissorRects(1, &scissorRect);

    //然后清除后台缓冲区和深度缓冲区，并赋值。步骤是先获得堆中描述符句柄（即地址），再通过ClearRenderTargetView函数和ClearDepthStencilView函数做清除和赋值。这里我们将RT资源背景色赋值为DarkRed（暗红）。
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(rtvHeap->GetCPUDescriptorHandleForHeapStart(), ref_mCurrentBackBuffer, rtvDescriptorSize);
    mCommandList->ClearRenderTargetView(rtvHandle, DirectX::Colors::DarkRed, 0, nullptr);//清除RT背景色为暗红，并且不设置裁剪矩形
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();
    mCommandList->ClearDepthStencilView(dsvHandle,	//DSV描述符句柄
        D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
        1.0f,	//默认深度值
        0,	//默认模板值
        0,	//裁剪矩形数量
        nullptr);	//裁剪矩形指针

    //然后我们指定将要渲染的缓冲区，即指定RTV和DSV。
    mCommandList->OMSetRenderTargets(1,//待绑定的RTV数量
        &rtvHandle,	//指向RTV数组的指针
        true,	//RTV对象在堆内存中是连续存放的
        &dsvHandle);	//指向DSV的指针

    //等到渲染完成，我们要将后台缓冲区的状态改成呈现状态，使其之后推到前台缓冲区显示。完了，关闭命令列表，等待传入命令队列。
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(swapChainBuffer[ref_mCurrentBackBuffer].Get(),
        D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));//从渲染目标到呈现
    //完成命令的记录关闭命令列表
    ThrowIfFailed(mCommandList->Close());

    //等CPU将命令都准备好后，需要将待执行的命令列表加入GPU的命令队列。使用的是ExecuteCommandLists函数。
    ID3D12CommandList* commandLists[] = { mCommandList.Get() };//声明并定义命令列表数组
    mCommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);//将命令从命令列表传至命令队列

    ThrowIfFailed(m_SwapChain->Present(0, 0));
    ref_mCurrentBackBuffer = (ref_mCurrentBackBuffer + 1) % 2;

    FlushCmdQueue();

    //beginScene();
    //render();
    //endScene();
}


#pragma region Initialize
bool QDirect3D12Widget::InitDirect3D()
{
    /*开启D3D12调试层*/
#if defined(DEBUG) || defined(_DEBUG)
    {
        ComPtr<ID3D12Debug> debugController;
        ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)));
        debugController->EnableDebugLayer();
    }
#endif

    CreateDevice();
    CreateFence();
    GetDescriptorSize();
    SetMSAA();
    CreateCommandObjects();
    CreateSwapChain();
    CreateDescriptorHeap();
    CreateRTV();
    CreateDSV();
    CreateViewPortAndScissorRect();

    return true;
}
/// <summary>
/// Initialize:: 1 Create the Device
/// </summary>
void QDirect3D12Widget::CreateDevice()
{
    DXCall(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
    DXCall(D3D12CreateDevice(nullptr, //此参数如果设置为nullptr，则使用主适配器
        D3D_FEATURE_LEVEL_12_0,		//应用程序需要硬件所支持的最低功能级别
        IID_PPV_ARGS(&md3dDevice)));	//返回所建设备
}
/// <summary>
/// Initialize:: 2 Create the Fance
/// </summary>
void QDirect3D12Widget::CreateFence()
{
    DXCall(md3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
}
/// <summary>
/// Initialize:: 3 Create Descriptor Sizes
/// </summary>
void QDirect3D12Widget::GetDescriptorSize()
{
    rtvDescriptorSize = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    dsvDescriptorSize = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    cbv_srv_uavDescriptorSize = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}
/// <summary>
/// Initialize:: 4 Check 4X MSAA Quality Support
/// </summary>
void QDirect3D12Widget::SetMSAA()
{
    msaaQualityLevels.Format = DXGI_FORMAT_R8G8B8A8_UNORM;	//UNORM是归一化处理的无符号整数
    msaaQualityLevels.SampleCount = 1;
    msaaQualityLevels.Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE;
    msaaQualityLevels.NumQualityLevels = 0;
    //当前图形驱动对MSAA多重采样的支持（注意：第二个参数即是输入又是输出）
    ThrowIfFailed(md3dDevice->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &msaaQualityLevels, sizeof(msaaQualityLevels)));
    //NumQualityLevels在Check函数里会进行设置
    //如果支持MSAA，则Check函数返回的NumQualityLevels > 0
    //expression为假（即为0），则终止程序运行，并打印一条出错信息
    assert(msaaQualityLevels.NumQualityLevels > 0);
}
/// <summary>
/// Initialize:: 5 Create Command Queue and Command Lists
/// </summary>
void QDirect3D12Widget::CreateCommandObjects()
{
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    DXCall(md3dDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&mCommandQueue)));

    DXCall(md3dDevice->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(mDirectCmdListAlloc.GetAddressOf())));

    DXCall(md3dDevice->CreateCommandList(
        0, //掩码值为0，单GPU
        D3D12_COMMAND_LIST_TYPE_DIRECT, //命令列表类型
        mDirectCmdListAlloc.Get(), // Associated command allocator	//命令分配器接口指针
        nullptr,                   // Initial PipelineStateObject	//流水线状态对象PSO，这里不绘制，所以空指针
        IID_PPV_ARGS(mCommandList.GetAddressOf())));	//返回创建的命令列表

    // Start off in a closed state.  This is because the first time we refer 
    // to the command list we will Reset it, and it needs to be closed before
    // calling Reset.
    mCommandList->Close();	//重置命令列表前必须将其关闭
}
/// <summary>
/// Initialize:: 6 Describe and Create Swap Chain
/// </summary>
void QDirect3D12Widget::CreateSwapChain()
{
    // Release the previous swapchain we will be recreating.
    m_SwapChain.Reset();

    DXGI_SWAP_CHAIN_DESC swapChainDesc;	//交换链描述结构体
    swapChainDesc.BufferDesc.Width = width();	//缓冲区分辨率的宽度
    swapChainDesc.BufferDesc.Height = height();	//缓冲区分辨率的高度
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;	//缓冲区的显示格式
    swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;	//刷新率的分子
    swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;	//刷新率的分母
    swapChainDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;	//逐行扫描VS隔行扫描(未指定的)
    swapChainDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;	//图像相对屏幕的拉伸（未指定的）
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;	//将数据渲染至后台缓冲区（即作为渲染目标）
    swapChainDesc.OutputWindow = (HWND)winId();	//渲染窗口句柄
    swapChainDesc.SampleDesc.Count = 1;	//多重采样数量
    swapChainDesc.SampleDesc.Quality = 0;	//多重采样质量
    swapChainDesc.Windowed = true;	//是否窗口化
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;	//固定写法
    swapChainDesc.BufferCount = 2;	//后台缓冲区数量（双缓冲）
    swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;	//自适应窗口模式（自动选择最适于当前窗口尺寸的显示模式）
    //利用DXGI接口下的工厂类创建交换链
    ThrowIfFailed(dxgiFactory->CreateSwapChain(mCommandQueue.Get(), &swapChainDesc, m_SwapChain.GetAddressOf()));
}
/// <summary>
/// Initialize:: 7 Create the Descriptor Heaps
/// </summary>
void QDirect3D12Widget::CreateDescriptorHeap()
{
    //首先创建RTV堆
    D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc;
    rtvDescriptorHeapDesc.NumDescriptors = 2;
    rtvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    rtvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDescriptorHeapDesc.NodeMask = 0;

    ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(&rtvHeap)));
    //然后创建DSV堆
    D3D12_DESCRIPTOR_HEAP_DESC dsvDescriptorHeapDesc;
    dsvDescriptorHeapDesc.NumDescriptors = 1;
    dsvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    dsvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvDescriptorHeapDesc.NodeMask = 0;
    ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&dsvDescriptorHeapDesc, IID_PPV_ARGS(&dsvHeap)));
}
/// <summary>
/// Initialize:: 8 Create Render Target View
/// </summary>
void QDirect3D12Widget::CreateRTV()
{
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHeapHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (int i = 0; i < 2; i++)
    {
        //获得存于交换链中的后台缓冲区资源
        m_SwapChain->GetBuffer(i, IID_PPV_ARGS(swapChainBuffer[i].GetAddressOf()));
        //创建RTV
        md3dDevice->CreateRenderTargetView(swapChainBuffer[i].Get(),
            nullptr,	//在交换链创建中已经定义了该资源的数据格式，所以这里指定为空指针
            rtvHeapHandle);	//描述符句柄结构体（这里是变体，继承自CD3DX12_CPU_DESCRIPTOR_HANDLE）
        //偏移到描述符堆中的下一个缓冲区
        rtvHeapHandle.Offset(1, rtvDescriptorSize);
    }
}
/// <summary>
/// Initialize:: 9 Create the Depth/Stencil Buffer & View
/// </summary>
void QDirect3D12Widget::CreateDSV()
{
    //在CPU中创建好深度模板数据资源
    D3D12_RESOURCE_DESC dsvResourceDesc;
    dsvResourceDesc.Alignment = 0;	//指定对齐
    dsvResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;	//指定资源维度（类型）为TEXTURE2D
    dsvResourceDesc.DepthOrArraySize = 1;	//纹理深度为1
    dsvResourceDesc.Width = 1280;	//资源宽
    dsvResourceDesc.Height = 720;	//资源高
    dsvResourceDesc.MipLevels = 1;	//MIPMAP层级数量
    dsvResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;	//指定纹理布局（这里不指定）
    dsvResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;	//深度模板资源的Flag
    dsvResourceDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;	//24位深度，8位模板,还有个无类型的格式DXGI_FORMAT_R24G8_TYPELESS也可以使用
    dsvResourceDesc.SampleDesc.Count = 4;	//多重采样数量
    dsvResourceDesc.SampleDesc.Quality = msaaQualityLevels.NumQualityLevels - 1;	//多重采样质量
    CD3DX12_CLEAR_VALUE optClear;	//清除资源的优化值，提高清除操作的执行速度（CreateCommittedResource函数中传入）
    optClear.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;//24位深度，8位模板,还有个无类型的格式DXGI_FORMAT_R24G8_TYPELESS也可以使用
    optClear.DepthStencil.Depth = 1;	//初始深度值为1
    optClear.DepthStencil.Stencil = 0;	//初始模板值为0
    //创建一个资源和一个堆，并将资源提交至堆中（将深度模板数据提交至GPU显存中）
    ThrowIfFailed(md3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),	//堆类型为默认堆（不能写入）
        D3D12_HEAP_FLAG_NONE,	//Flag
        &dsvResourceDesc,	//上面定义的DSV资源指针
        D3D12_RESOURCE_STATE_COMMON,	//资源的状态为初始状态
        &optClear,	//上面定义的优化值指针
        IID_PPV_ARGS(&depthStencilBuffer)));	//返回深度模板资源
        //创建DSV(必须填充DSV属性结构体，和创建RTV不同，RTV是通过句柄)
        //D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc;
        //dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
        //dsvDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        //dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        //dsvDesc.Texture2D.MipSlice = 0;
    md3dDevice->CreateDepthStencilView(depthStencilBuffer.Get(),
        nullptr,	//D3D12_DEPTH_STENCIL_VIEW_DESC类型指针，可填&dsvDesc（见上注释代码），
                            //由于在创建深度模板资源时已经定义深度模板数据属性，所以这里可以指定为空指针
        dsvHeap->GetCPUDescriptorHandleForHeapStart());	//DSV句柄

    //// Transition the resource from its initial state to be used as a depth buffer.
    //mCommandList->ResourceBarrier(1,	//Barrier屏障个数
    //    &CD3DX12_RESOURCE_BARRIER::Transition(depthStencilBuffer.Get(),
    //        D3D12_RESOURCE_STATE_COMMON,	//转换前状态（创建时的状态，即CreateCommittedResource函数中定义的状态）
    //        D3D12_RESOURCE_STATE_DEPTH_WRITE));

    ////等所有命令都进入cmdList后，还需要用ExecuteCommandLists函数，将命令从命令列表传入命令队列，也就是从CPU传入GPU的过程。注意：在传入命令队列前必须关闭命令列表。
    //ThrowIfFailed(mCommandList->Close());	//命令添加完后将其关闭
    //ID3D12CommandList* cmdLists[] = { mCommandList.Get() };	//声明并定义命令列表数组
    //mCommandQueue->ExecuteCommandLists(_countof(cmdLists), cmdLists);	//将命令从命令列表传至命令队列
}
/// <summary>
/// Initialize:: 11 Create the Depth/Stencil Buffer & View
/// </summary>
void QDirect3D12Widget::CreateViewPortAndScissorRect()
{
    //视口设置
    viewPort.TopLeftX = 0;
    viewPort.TopLeftY = 0;
    viewPort.Width = 1280;
    viewPort.Height = 720;
    viewPort.MaxDepth = 1.0f;
    viewPort.MinDepth = 0.0f;
    //裁剪矩形设置（矩形外的像素都将被剔除）
    //前两个为左上点坐标，后两个为右下点坐标
    scissorRect.left = 0;
    scissorRect.top = 0;
    scissorRect.right = 1280;
    scissorRect.bottom = 720;
}
#pragma endregion

#pragma region Deprecated
//void QDirect3D12Widget::create3DDevice()
//{
//    // 1. Create ID3D12Device
//    // ==================================================
//    UINT factoryFlags = 0;
//
//#ifdef _DEBUG
//    {
//        ComPtr<ID3D12Debug> dx12Debug;
//        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dx12Debug))))
//        {
//            dx12Debug->EnableDebugLayer();
//
//            // Enable additional debug layers.
//            factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
//        }
//    }
//#endif
//
//    DXCall(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&m_pFactory)));
//
//    // Try and get hardware adapter compatible with d3d12, if not found, use wrap.
//    ComPtr<IDXGIAdapter1> adapter;
//    getHardwareAdapter(m_pFactory, adapter.GetAddressOf());
//    if (!adapter) DXCall(m_pFactory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
//
//    DXCall(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_pDevice)));
//
//    // Describe and create the command queue.
//    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
//    queueDesc.Flags                    = D3D12_COMMAND_QUEUE_FLAG_NONE;
//    queueDesc.Type                     = D3D12_COMMAND_LIST_TYPE_DIRECT;
//    DXCall(m_pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_pCommandQueue)));
//
//    // Describe and create the swap chain.
//    {
//        DXGI_SWAP_CHAIN_DESC1 sd = {};
//        sd.BufferCount           = FRAME_COUNT;
//        sd.Width                 = width();
//        sd.Height                = height();
//        sd.Format                = DXGI_FORMAT_R8G8B8A8_UNORM;
//        // sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
//        sd.Flags              = 0;
//        sd.BufferUsage        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
//        sd.SampleDesc.Count   = 1;
//        sd.SampleDesc.Quality = 0;
//        sd.SwapEffect         = DXGI_SWAP_EFFECT_FLIP_DISCARD;
//        sd.AlphaMode          = DXGI_ALPHA_MODE_UNSPECIFIED;
//        sd.Scaling            = DXGI_SCALING_NONE;
//        sd.Stereo             = FALSE;
//
//        DXGI_SWAP_CHAIN_FULLSCREEN_DESC fsSd = {};
//        fsSd.Windowed                        = TRUE;
//
//        ComPtr<IDXGISwapChain1> swapChain1;
//        DXCall(m_pFactory->CreateSwapChainForHwnd(m_pCommandQueue, m_hWnd, &sd, &fsSd,
//                                                  Q_NULLPTR, swapChain1.GetAddressOf()));
//        DXCall(swapChain1->QueryInterface(IID_PPV_ARGS(&m_pSwapChain)));
//        m_iCurrFrameIndex = m_pSwapChain->GetCurrentBackBufferIndex();
//    }
//
//    // Create render target view(RTV) descriptor heaps and handles.
//    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc = {};
//    rtvDesc.NumDescriptors             = FRAME_COUNT;
//    rtvDesc.Type                       = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
//    rtvDesc.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
//    DXCall(m_pDevice->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&m_pRTVDescHeap)));
//    m_iRTVDescSize =
//        m_pDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
//
//    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(
//        m_pRTVDescHeap->GetCPUDescriptorHandleForHeapStart());
//    for (UINT i = 0; i < FRAME_COUNT; i++)
//    {
//        m_RTVDescriptors[i] = rtvHandle;
//        DXCall(m_pSwapChain->GetBuffer(i, IID_PPV_ARGS(&m_pRTVResources[i])));
//        m_pDevice->CreateRenderTargetView(m_pRTVResources[i], Q_NULLPTR, m_RTVDescriptors[i]);
//        rtvHandle.Offset(1, m_iRTVDescSize);
//    }
//
//    // Create shader resource view(SRV) descriptor heap.
//    D3D12_DESCRIPTOR_HEAP_DESC srvDesc = {};
//    srvDesc.NumDescriptors             = 1;
//    srvDesc.Type                       = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
//    srvDesc.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
//    DXCall(m_pDevice->CreateDescriptorHeap(&srvDesc, IID_PPV_ARGS(&m_pSrvDescHeap)));
//
//    // Create command allocator for each frame.
//    for (UINT i = 0; i < FRAME_COUNT; i++)
//    {
//        DXCall(m_pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
//                                                 IID_PPV_ARGS(&m_pCommandAllocators[i])));
//    }
//
//    // Create command list. We don't create PSO here, so we set it to Q_NULLPTR to use the
//    // default PSO. Command list by default set on recording state when created, therefore we
//    // close it for now.
//    DXCall(m_pDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
//                                        m_pCommandAllocators[m_iCurrFrameIndex], Q_NULLPTR,
//                                        IID_PPV_ARGS(&m_pCommandList)));
//    DXCall(m_pCommandList->Close());
//
//    // 2. Create the Fence and Descriptor Sizes
//    // ==================================================
//    // Create synchronized objects.
//    DXCall(m_pDevice->CreateFence(m_iFenceValues[m_iCurrFrameIndex], D3D12_FENCE_FLAG_NONE,
//                                  IID_PPV_ARGS(&m_pFence)));
//    m_iFenceValues[m_iCurrFrameIndex]++;
//
//    m_hFenceEvent = CreateEvent(Q_NULLPTR, FALSE, FALSE, Q_NULLPTR);
//    if (!m_hFenceEvent) DXCall(HRESULT_FROM_WIN32(GetLastError()));
//
//    // DXCall(m_pSwapChain->SetMaximumFrameLatency(FRAME_COUNT));
//    // m_hSwapChainEvent = m_pSwapChain->GetFrameLatencyWaitableObject();
//
//    // Wait for the GPU to complete our setup before proceeding.
//    waitForGpu();
//}
#pragma endregion

void QDirect3D12Widget::beginScene()
{
    DXCall(m_pCommandAllocators[m_iCurrFrameIndex]->Reset());
    DXCall(m_pCommandList->Reset(m_pCommandAllocators[m_iCurrFrameIndex], Q_NULLPTR));

    m_pCommandList->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::Transition(m_pRTVResources[m_iCurrFrameIndex],
                                                 D3D12_RESOURCE_STATE_PRESENT,
                                                 D3D12_RESOURCE_STATE_RENDER_TARGET));
}

void QDirect3D12Widget::endScene()
{
    m_pCommandList->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::Transition(m_pRTVResources[m_iCurrFrameIndex],
                                                 D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                 D3D12_RESOURCE_STATE_PRESENT));

    DXCall(m_pCommandList->Close());
    m_pCommandQueue->ExecuteCommandLists(
        1, reinterpret_cast<ID3D12CommandList * const *>(&m_pCommandList));

    DXCall(m_pSwapChain->Present(1, 0));

    moveToNextFrame();
}

void QDirect3D12Widget::tick()
{
    // TODO: Update your scene here. For aesthetics reasons, only do it here if it's an
    // important component, otherwise do it in the MainWindow.
    // m_pCamera->Tick();

    emit ticked();
}

void QDirect3D12Widget::render()
{
    // Start recording the render commands
    m_pCommandList->ClearRenderTargetView(m_RTVDescriptors[m_iCurrFrameIndex],
                                          reinterpret_cast<const float *>(&m_BackColor), 0,
                                          Q_NULLPTR);
    m_pCommandList->OMSetRenderTargets(1, &m_RTVDescriptors[m_iCurrFrameIndex], FALSE,
                                       Q_NULLPTR);
    m_pCommandList->SetDescriptorHeaps(1, &m_pSrvDescHeap);

    // TODO: Present your scene here. For aesthetics reasons, only do it here if it's an
    // important component, otherwise do it in the MainWindow.
    // m_pCamera->Apply();

    emit rendered(m_pCommandList);
}

void QDirect3D12Widget::onReset()
{
    // TODO(Gilad): FIXME: this needs to be done in a synchronized manner. Need to look at
    // DirectX-12 samples here: https://github.com/microsoft/DirectX-Graphics-Samples how to
    // properly do this without leaking memory.
    pauseFrames();
    resizeSwapChain(width(), height());
    continueFrames();
}

void QDirect3D12Widget::cleanupRenderTarget()
{
    waitForGpu();

    for (UINT i = 0; i < FRAME_COUNT; i++)
    {
        ReleaseObject(m_pRTVResources[i]);
        m_iFenceValues[i] = m_iFenceValues[m_iCurrFrameIndex];
    }
}

void QDirect3D12Widget::createRenderTarget()
{
    for (UINT i = 0; i < FRAME_COUNT; i++)
    {
        DXCall(m_pSwapChain->GetBuffer(i, IID_PPV_ARGS(&m_pRTVResources[i])));
        m_pDevice->CreateRenderTargetView(m_pRTVResources[i], Q_NULLPTR, m_RTVDescriptors[i]);
    }
}

void QDirect3D12Widget::waitForGpu()
{
    DXCall(m_pCommandQueue->Signal(m_pFence, m_iFenceValues[m_iCurrFrameIndex]));

    DXCall(m_pFence->SetEventOnCompletion(m_iFenceValues[m_iCurrFrameIndex], m_hFenceEvent));
    WaitForSingleObject(m_hFenceEvent, INFINITE);

    m_iFenceValues[m_iCurrFrameIndex]++;
}

void QDirect3D12Widget::moveToNextFrame()
{
    const UINT64 currentFenceValue = m_iFenceValues[m_iCurrFrameIndex];
    DXCall(m_pCommandQueue->Signal(m_pFence, currentFenceValue));

    m_iCurrFrameIndex = m_pSwapChain->GetCurrentBackBufferIndex();
    if (m_pFence->GetCompletedValue() < m_iFenceValues[m_iCurrFrameIndex])
    {
        DXCall(
            m_pFence->SetEventOnCompletion(m_iFenceValues[m_iCurrFrameIndex], m_hFenceEvent));
        WaitForSingleObject(m_hFenceEvent, INFINITE);
    }

    m_iFenceValues[m_iCurrFrameIndex] = currentFenceValue + 1;
}

void QDirect3D12Widget::resizeSwapChain(int width, int height)
{
    // ReleaseHandle(m_hSwapChainEvent);
    cleanupRenderTarget();

    if (m_pSwapChain)
    {
        DXCall(m_pSwapChain->ResizeBuffers(FRAME_COUNT, width, height,
                                           DXGI_FORMAT_R8G8B8A8_UNORM, 0));
    }
    else
    {
        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.BufferCount           = FRAME_COUNT;
        sd.Width                 = width;
        sd.Height                = height;
        sd.Format                = DXGI_FORMAT_R8G8B8A8_UNORM;
        // sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        sd.Flags              = 0;
        sd.BufferUsage        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.SampleDesc.Count   = 1;
        sd.SampleDesc.Quality = 0;
        sd.SwapEffect         = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.AlphaMode          = DXGI_ALPHA_MODE_UNSPECIFIED;
        sd.Scaling            = DXGI_SCALING_NONE;
        sd.Stereo             = FALSE;

        DXGI_SWAP_CHAIN_FULLSCREEN_DESC fsSd = {};
        fsSd.Windowed                        = TRUE;

        ComPtr<IDXGISwapChain1> m_SwapChain;
        DXCall(m_pFactory->CreateSwapChainForHwnd(m_pCommandQueue, m_hWnd, &sd, &fsSd,
                                                  Q_NULLPTR, m_SwapChain.GetAddressOf()));
        DXCall(m_SwapChain->QueryInterface(IID_PPV_ARGS(&m_pSwapChain)));

        // DXCall(m_pSwapChain->SetMaximumFrameLatency(FRAME_COUNT));
        // m_hSwapChainEvent = m_pSwapChain->GetFrameLatencyWaitableObject();
    }

    createRenderTarget();

    m_iCurrFrameIndex = m_pSwapChain->GetCurrentBackBufferIndex();
}

void QDirect3D12Widget::getHardwareAdapter(IDXGIFactory2 *  pFactory,
                                           IDXGIAdapter1 ** ppAdapter)
{
    ComPtr<IDXGIAdapter1> adapter;
    *ppAdapter = Q_NULLPTR;

    for (UINT adapterIndex = 0;
         DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(adapterIndex, &adapter);
         ++adapterIndex)
    {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        // Skip software adapter.
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

        // Check to see if the adapter supports Direct3D 12, but don't create the actual device
        // yet.
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                        _uuidof(ID3D12Device), Q_NULLPTR)))
            break;
    }

    *ppAdapter = adapter.Detach();
}

void QDirect3D12Widget::resetEnvironment()
{
    // TODO: Add your own custom default environment, i.e:
    // m_pCamera->resetCamera();

    onReset();

    if (!m_bRenderActive) tick();
}

void QDirect3D12Widget::wheelEvent(QWheelEvent * event)
{
    if (event->angleDelta().x() == 0)
    {
        // TODO: Update your camera position based on the delta value.
    }
    else if (event->angleDelta().x() !=
             0) // horizontal scrolling - mice with another side scroller.
    {
        // MouseWheelH += (float)(event->angleDelta().y() / WHEEL_DELTA);
    }
    else if (event->angleDelta().y() != 0)
    {
        // MouseWheel += (float)(event->angleDelta().y() / WHEEL_DELTA);
    }

    QWidget::wheelEvent(event);
}

QPaintEngine * QDirect3D12Widget::paintEngine() const
{
    return Q_NULLPTR;
}

void QDirect3D12Widget::paintEvent(QPaintEvent * event) {}

void QDirect3D12Widget::resizeEvent(QResizeEvent * event)
{
    //if (m_bDeviceInitialized)
    //{
    //    //Debug Change
    //    onReset();
    //    emit widgetResized();
    //}

    QWidget::resizeEvent(event);
}

bool QDirect3D12Widget::event(QEvent * event)
{
    switch (event->type())
    {
        // Workaround for https://bugreports.qt.io/browse/QTBUG-42183 to get key strokes.
        // To make sure that we always have focus on the widget when we enter the rect area.
        case QEvent::Enter:
        case QEvent::FocusIn:
        case QEvent::FocusAboutToChange:
            if (::GetFocus() != m_hWnd)
            {
                QWidget * nativeParent = this;
                while (true)
                {
                    if (nativeParent->isWindow()) break;

                    QWidget * parent = nativeParent->nativeParentWidget();
                    if (!parent) break;

                    nativeParent = parent;
                }

                if (nativeParent && nativeParent != this &&
                    ::GetFocus() == reinterpret_cast<HWND>(nativeParent->winId()))
                    ::SetFocus(m_hWnd);
            }
            break;
        case QEvent::KeyPress:
            emit keyPressed((QKeyEvent *)event);
            break;
        case QEvent::MouseMove:
            emit mouseMoved((QMouseEvent *)event);
            break;
        case QEvent::MouseButtonPress:
            emit mouseClicked((QMouseEvent *)event);
            break;
        case QEvent::MouseButtonRelease:
            emit mouseReleased((QMouseEvent *)event);
            break;
    }

    return QWidget::event(event);
}

LRESULT QDirect3D12Widget::WndProc(MSG * pMsg)
{
    // Process wheel events using Qt's event-system.
    if (pMsg->message == WM_MOUSEWHEEL || pMsg->message == WM_MOUSEHWHEEL) return false;

    return false;
}

#if QT_VERSION >= 0x050000
bool QDirect3D12Widget::nativeEvent(const QByteArray & eventType,
                                    void *             message,
                                    long *             result)
{
    Q_UNUSED(eventType);
    Q_UNUSED(result);

#    ifdef Q_OS_WIN
    MSG * pMsg = reinterpret_cast<MSG *>(message);
    return WndProc(pMsg);
#    endif

    return QWidget::nativeEvent(eventType, message, result);
}

#else // QT_VERSION < 0x050000
bool QDirect3D12Widget::winEvent(MSG * message, long * result)
{
    Q_UNUSED(result);

#    ifdef Q_OS_WIN
    MSG * pMsg = reinterpret_cast<MSG *>(message);
    return WndProc(pMsg);
#    endif

    return QWidget::winEvent(message, result);
}
#endif // QT_VERSION >= 0x050000
