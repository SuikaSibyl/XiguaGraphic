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
    mCurrentFence++;	//CPU��������رպ󣬽���ǰΧ��ֵ+1
    mCommandQueue->Signal(fence.Get(), mCurrentFence);	//��GPU������CPU���������󣬽�fence�ӿ��е�Χ��ֵ+1����fence->GetCompletedValue()+1
    if (fence->GetCompletedValue() < mCurrentFence)	//���С�ڣ�˵��GPUû�д�������������
    {
        HANDLE eventHandle = CreateEvent(nullptr, false, false, L"FenceSetDone");	//�����¼�
        fence->SetEventOnCompletion(mCurrentFence, eventHandle);//��Χ���ﵽmCurrentFenceֵ����ִ�е�Signal����ָ���޸���Χ��ֵ��ʱ������eventHandle�¼�
        WaitForSingleObject(eventHandle, INFINITE);//�ȴ�GPU����Χ���������¼���������ǰ�߳�ֱ���¼�������ע���Enent���������ٵȴ���
                               //���û��Set��Wait���������ˣ�Set��Զ������ã�����Ҳ��û�߳̿��Ի�������̣߳�
        CloseHandle(eventHandle);
    }
}
/// <summary>
/// 
/// </summary>
void QDirect3D12Widget::CalculateFrameState()
{
    static int frameCnt = 0;	//��֡��
    static float timeElapsed = 0.0f;	//���ŵ�ʱ��
    frameCnt++;	//ÿ֡++������һ����伴ΪFPSֵ
    //����ģ��
    /*std::wstring text = std::to_wstring(gt.TotalTime());
    std::wstring windowText = text;
    SetWindowText(mhMainWnd, windowText.c_str());*/
    //�ж�ģ��
    if (timer.TotalTime() - timeElapsed >= 1.0f)	//һ��>=0��˵���պù�һ��
    {
        float fps = (float)frameCnt;//ÿ�����֡
        float mspf = 1000.0f / fps;	//ÿ֡���ٺ���

        std::wstring fpsStr = std::to_wstring(fps);//תΪ���ַ�
        std::wstring mspfStr = std::to_wstring(mspf);
        //��֡������ʾ�ڴ�����
        std::wstring windowText = L"D3D12Init    fps:" + fpsStr + L"    " + L"mspf" + mspfStr;
        SetWindowText((HWND)winId(), windowText.c_str());

        //Ϊ������һ��֡��ֵ������
        frameCnt = 0;
        timeElapsed += 1.0f;
    }
}
void QDirect3D12Widget::Draw()
{
    //�����������������cmdAllocator�������б�cmdList��Ŀ��������������б���������ڴ档
    ThrowIfFailed(mDirectCmdListAlloc->Reset());//�ظ�ʹ�ü�¼���������ڴ�
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));//���������б����ڴ�

    //�������ǽ���̨������Դ�ӳ���״̬ת������ȾĿ��״̬����׼������ͼ����Ⱦ����
    UINT& ref_mCurrentBackBuffer = mCurrentBackBuffer;
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(swapChainBuffer[ref_mCurrentBackBuffer].Get(),//ת����ԴΪ��̨��������Դ
        D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));//�ӳ��ֵ���ȾĿ��ת��

    //�����������ӿںͲü����Ρ�
    mCommandList->RSSetViewports(1, &viewPort);
    mCommandList->RSSetScissorRects(1, &scissorRect);

    //Ȼ�������̨����������Ȼ�����������ֵ���������Ȼ�ö������������������ַ������ͨ��ClearRenderTargetView������ClearDepthStencilView����������͸�ֵ���������ǽ�RT��Դ����ɫ��ֵΪDarkRed�����죩��
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(rtvHeap->GetCPUDescriptorHandleForHeapStart(), ref_mCurrentBackBuffer, rtvDescriptorSize);
    mCommandList->ClearRenderTargetView(rtvHandle, DirectX::Colors::DarkRed, 0, nullptr);//���RT����ɫΪ���죬���Ҳ����òü�����
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();
    mCommandList->ClearDepthStencilView(dsvHandle,	//DSV���������
        D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
        1.0f,	//Ĭ�����ֵ
        0,	//Ĭ��ģ��ֵ
        0,	//�ü���������
        nullptr);	//�ü�����ָ��

    //Ȼ������ָ����Ҫ��Ⱦ�Ļ���������ָ��RTV��DSV��
    mCommandList->OMSetRenderTargets(1,//���󶨵�RTV����
        &rtvHandle,	//ָ��RTV�����ָ��
        true,	//RTV�����ڶ��ڴ�����������ŵ�
        &dsvHandle);	//ָ��DSV��ָ��

    //�ȵ���Ⱦ��ɣ�����Ҫ����̨��������״̬�ĳɳ���״̬��ʹ��֮���Ƶ�ǰ̨��������ʾ�����ˣ��ر������б��ȴ�����������С�
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(swapChainBuffer[ref_mCurrentBackBuffer].Get(),
        D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));//����ȾĿ�굽����
    //�������ļ�¼�ر������б�
    ThrowIfFailed(mCommandList->Close());

    //��CPU�����׼���ú���Ҫ����ִ�е������б����GPU��������С�ʹ�õ���ExecuteCommandLists������
    ID3D12CommandList* commandLists[] = { mCommandList.Get() };//���������������б�����
    mCommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);//������������б����������

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
    /*����D3D12���Բ�*/
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
    DXCall(D3D12CreateDevice(nullptr, //�˲����������Ϊnullptr����ʹ����������
        D3D_FEATURE_LEVEL_12_0,		//Ӧ�ó�����ҪӲ����֧�ֵ���͹��ܼ���
        IID_PPV_ARGS(&md3dDevice)));	//���������豸
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
    msaaQualityLevels.Format = DXGI_FORMAT_R8G8B8A8_UNORM;	//UNORM�ǹ�һ��������޷�������
    msaaQualityLevels.SampleCount = 1;
    msaaQualityLevels.Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE;
    msaaQualityLevels.NumQualityLevels = 0;
    //��ǰͼ��������MSAA���ز�����֧�֣�ע�⣺�ڶ������������������������
    ThrowIfFailed(md3dDevice->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &msaaQualityLevels, sizeof(msaaQualityLevels)));
    //NumQualityLevels��Check��������������
    //���֧��MSAA����Check�������ص�NumQualityLevels > 0
    //expressionΪ�٣���Ϊ0��������ֹ�������У�����ӡһ��������Ϣ
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
        0, //����ֵΪ0����GPU
        D3D12_COMMAND_LIST_TYPE_DIRECT, //�����б�����
        mDirectCmdListAlloc.Get(), // Associated command allocator	//����������ӿ�ָ��
        nullptr,                   // Initial PipelineStateObject	//��ˮ��״̬����PSO�����ﲻ���ƣ����Կ�ָ��
        IID_PPV_ARGS(mCommandList.GetAddressOf())));	//���ش����������б�

    // Start off in a closed state.  This is because the first time we refer 
    // to the command list we will Reset it, and it needs to be closed before
    // calling Reset.
    mCommandList->Close();	//���������б�ǰ���뽫��ر�
}
/// <summary>
/// Initialize:: 6 Describe and Create Swap Chain
/// </summary>
void QDirect3D12Widget::CreateSwapChain()
{
    // Release the previous swapchain we will be recreating.
    m_SwapChain.Reset();

    DXGI_SWAP_CHAIN_DESC swapChainDesc;	//�����������ṹ��
    swapChainDesc.BufferDesc.Width = width();	//�������ֱ��ʵĿ��
    swapChainDesc.BufferDesc.Height = height();	//�������ֱ��ʵĸ߶�
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;	//����������ʾ��ʽ
    swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;	//ˢ���ʵķ���
    swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;	//ˢ���ʵķ�ĸ
    swapChainDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;	//����ɨ��VS����ɨ��(δָ����)
    swapChainDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;	//ͼ�������Ļ�����죨δָ���ģ�
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;	//��������Ⱦ����̨������������Ϊ��ȾĿ�꣩
    swapChainDesc.OutputWindow = (HWND)winId();	//��Ⱦ���ھ��
    swapChainDesc.SampleDesc.Count = 1;	//���ز�������
    swapChainDesc.SampleDesc.Quality = 0;	//���ز�������
    swapChainDesc.Windowed = true;	//�Ƿ񴰿ڻ�
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;	//�̶�д��
    swapChainDesc.BufferCount = 2;	//��̨������������˫���壩
    swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;	//����Ӧ����ģʽ���Զ�ѡ�������ڵ�ǰ���ڳߴ����ʾģʽ��
    //����DXGI�ӿ��µĹ����ഴ��������
    ThrowIfFailed(dxgiFactory->CreateSwapChain(mCommandQueue.Get(), &swapChainDesc, m_SwapChain.GetAddressOf()));
}
/// <summary>
/// Initialize:: 7 Create the Descriptor Heaps
/// </summary>
void QDirect3D12Widget::CreateDescriptorHeap()
{
    //���ȴ���RTV��
    D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc;
    rtvDescriptorHeapDesc.NumDescriptors = 2;
    rtvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    rtvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDescriptorHeapDesc.NodeMask = 0;

    ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(&rtvHeap)));
    //Ȼ�󴴽�DSV��
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
        //��ô��ڽ������еĺ�̨��������Դ
        m_SwapChain->GetBuffer(i, IID_PPV_ARGS(swapChainBuffer[i].GetAddressOf()));
        //����RTV
        md3dDevice->CreateRenderTargetView(swapChainBuffer[i].Get(),
            nullptr,	//�ڽ������������Ѿ������˸���Դ�����ݸ�ʽ����������ָ��Ϊ��ָ��
            rtvHeapHandle);	//����������ṹ�壨�����Ǳ��壬�̳���CD3DX12_CPU_DESCRIPTOR_HANDLE��
        //ƫ�Ƶ����������е���һ��������
        rtvHeapHandle.Offset(1, rtvDescriptorSize);
    }
}
/// <summary>
/// Initialize:: 9 Create the Depth/Stencil Buffer & View
/// </summary>
void QDirect3D12Widget::CreateDSV()
{
    //��CPU�д��������ģ��������Դ
    D3D12_RESOURCE_DESC dsvResourceDesc;
    dsvResourceDesc.Alignment = 0;	//ָ������
    dsvResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;	//ָ����Դά�ȣ����ͣ�ΪTEXTURE2D
    dsvResourceDesc.DepthOrArraySize = 1;	//�������Ϊ1
    dsvResourceDesc.Width = 1280;	//��Դ��
    dsvResourceDesc.Height = 720;	//��Դ��
    dsvResourceDesc.MipLevels = 1;	//MIPMAP�㼶����
    dsvResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;	//ָ�������֣����ﲻָ����
    dsvResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;	//���ģ����Դ��Flag
    dsvResourceDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;	//24λ��ȣ�8λģ��,���и������͵ĸ�ʽDXGI_FORMAT_R24G8_TYPELESSҲ����ʹ��
    dsvResourceDesc.SampleDesc.Count = 4;	//���ز�������
    dsvResourceDesc.SampleDesc.Quality = msaaQualityLevels.NumQualityLevels - 1;	//���ز�������
    CD3DX12_CLEAR_VALUE optClear;	//�����Դ���Ż�ֵ��������������ִ���ٶȣ�CreateCommittedResource�����д��룩
    optClear.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;//24λ��ȣ�8λģ��,���и������͵ĸ�ʽDXGI_FORMAT_R24G8_TYPELESSҲ����ʹ��
    optClear.DepthStencil.Depth = 1;	//��ʼ���ֵΪ1
    optClear.DepthStencil.Stencil = 0;	//��ʼģ��ֵΪ0
    //����һ����Դ��һ���ѣ�������Դ�ύ�����У������ģ�������ύ��GPU�Դ��У�
    ThrowIfFailed(md3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),	//������ΪĬ�϶ѣ�����д�룩
        D3D12_HEAP_FLAG_NONE,	//Flag
        &dsvResourceDesc,	//���涨���DSV��Դָ��
        D3D12_RESOURCE_STATE_COMMON,	//��Դ��״̬Ϊ��ʼ״̬
        &optClear,	//���涨����Ż�ֵָ��
        IID_PPV_ARGS(&depthStencilBuffer)));	//�������ģ����Դ
        //����DSV(�������DSV���Խṹ�壬�ʹ���RTV��ͬ��RTV��ͨ�����)
        //D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc;
        //dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
        //dsvDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        //dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        //dsvDesc.Texture2D.MipSlice = 0;
    md3dDevice->CreateDepthStencilView(depthStencilBuffer.Get(),
        nullptr,	//D3D12_DEPTH_STENCIL_VIEW_DESC����ָ�룬����&dsvDesc������ע�ʹ��룩��
                            //�����ڴ������ģ����Դʱ�Ѿ��������ģ���������ԣ������������ָ��Ϊ��ָ��
        dsvHeap->GetCPUDescriptorHandleForHeapStart());	//DSV���

    //// Transition the resource from its initial state to be used as a depth buffer.
    //mCommandList->ResourceBarrier(1,	//Barrier���ϸ���
    //    &CD3DX12_RESOURCE_BARRIER::Transition(depthStencilBuffer.Get(),
    //        D3D12_RESOURCE_STATE_COMMON,	//ת��ǰ״̬������ʱ��״̬����CreateCommittedResource�����ж����״̬��
    //        D3D12_RESOURCE_STATE_DEPTH_WRITE));

    ////�������������cmdList�󣬻���Ҫ��ExecuteCommandLists������������������б���������У�Ҳ���Ǵ�CPU����GPU�Ĺ��̡�ע�⣺�ڴ����������ǰ����ر������б�
    //ThrowIfFailed(mCommandList->Close());	//������������ر�
    //ID3D12CommandList* cmdLists[] = { mCommandList.Get() };	//���������������б�����
    //mCommandQueue->ExecuteCommandLists(_countof(cmdLists), cmdLists);	//������������б����������
}
/// <summary>
/// Initialize:: 11 Create the Depth/Stencil Buffer & View
/// </summary>
void QDirect3D12Widget::CreateViewPortAndScissorRect()
{
    //�ӿ�����
    viewPort.TopLeftX = 0;
    viewPort.TopLeftY = 0;
    viewPort.Width = 1280;
    viewPort.Height = 720;
    viewPort.MaxDepth = 1.0f;
    viewPort.MinDepth = 0.0f;
    //�ü��������ã�����������ض������޳���
    //ǰ����Ϊ���ϵ����꣬������Ϊ���µ�����
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
