#include "QDirect3D12Widget.h"
#include <DirectXColors.h>

#include <array>
#include <QDebug>
#include <QEvent>
#include <QWheelEvent>
#include <DirectXMath.h>

#include <Shader.h>
#include <UploadBuffer.h>

using Microsoft::WRL::ComPtr;
using namespace DirectX;

constexpr int FPS_LIMIT    = 120.0f;
constexpr int MS_PER_FRAME = (int)((1.0f / FPS_LIMIT) * 1000.0f);

//定义顶点结构体
struct Vertex
{
    XMFLOAT3 Pos;
    XMFLOAT4 Color;
};

QDirect3D12Widget::QDirect3D12Widget(QWidget * parent)
    : QWidget(parent)
    , m_hWnd(reinterpret_cast<HWND>(winId()))
    , m_bDeviceInitialized(false)
    , m_bRenderActive(false)
    , m_bStarted(false)
{
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
    timer.Stop();
    m_bRenderActive = false;
}

void QDirect3D12Widget::continueFrames()
{
    if (m_qTimer.isActive() || !m_bStarted) return;

    connect(&m_qTimer, &QTimer::timeout, this, &QDirect3D12Widget::onFrame);
    m_qTimer.start(MS_PER_FRAME);
    timer.Start();
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
void QDirect3D12Widget::BuildDescriptorHeaps()
{
    D3D12_DESCRIPTOR_HEAP_DESC cbvHeapDesc;
    cbvHeapDesc.NumDescriptors = 1;
    cbvHeapDesc.Type =
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvHeapDesc.Flags =
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    cbvHeapDesc.NodeMask = 0;
    ThrowIfFailed(md3dDevice -> CreateDescriptorHeap(&cbvHeapDesc,
        IID_PPV_ARGS(&mCbvHeap)));
}

void QDirect3D12Widget::BuildConstantBuffers()
{
    mObjectCB = std::make_unique<UploadBuffer<ObjectConstants>>(md3dDevice.Get(), 1, true);
    UINT objCBByteSize =
        Utils::CalcConstantBufferByteSize(sizeof(ObjectConstants));
    D3D12_GPU_VIRTUAL_ADDRESS cbAddress = mObjectCB -> Resource()->GetGPUVirtualAddress();
    // Offset to the ith object constant buffer in the buffer.
        // Here our i = 0.
        int boxCBufIndex = 0;
    cbAddress += boxCBufIndex * objCBByteSize;
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
    cbvDesc.BufferLocation = cbAddress;
    cbvDesc.SizeInBytes =
        Utils::CalcConstantBufferByteSize(sizeof(ObjectConstants));
    md3dDevice->CreateConstantBufferView(
        &cbvDesc,
        mCbvHeap->GetCPUDescriptorHandleForHeapStart());
}

void QDirect3D12Widget::BuildRootSignature()
{
    // Shader programs typically require resources as input(constant
    // buffers, textures, samplers). The root signature defines the
    // resources the shader programs expect. If we think of the shader
    // programs as a function, and the input resources as function
    // parameters, then the root signature can be thought of as defining
    // the function signature.
    // Root parameter can be a table, root descriptor or root constants.
    CD3DX12_ROOT_PARAMETER slotRootParameter[1];
    // Create a single descriptor table of CBVs.
    CD3DX12_DESCRIPTOR_RANGE cbvTable;
    cbvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1,
        0);
    slotRootParameter[0].InitAsDescriptorTable(1,
        &cbvTable);
    // A root signature is an array of root parameters.
    CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(1,
        slotRootParameter, 0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
    // create a root signature with a single slot which points to a
    // descriptor range consisting of a single constant buffer
    ComPtr<ID3DBlob> serializedRootSig = nullptr;
    ComPtr<ID3DBlob> errorBlob = nullptr;
    HRESULT hr =
        D3D12SerializeRootSignature(&rootSigDesc,
            D3D_ROOT_SIGNATURE_VERSION_1,
            serializedRootSig.GetAddressOf(),
            errorBlob.GetAddressOf());
    if (errorBlob != nullptr)
    {
        ::OutputDebugStringA((char*)errorBlob -> GetBufferPointer());
    }
    ThrowIfFailed(hr);
    ThrowIfFailed(md3dDevice->CreateRootSignature(
        0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(&mRootSignature)));
}
void QDirect3D12Widget::BuildShadersAndInputLayout()
{
    mpShader = new Shader(md3dDevice);
}
void QDirect3D12Widget::BuildBoxGeometry()
{
    std::array<Vertex, 8> vertices =
    {
    Vertex({ XMFLOAT3(-1.0f, -1.0f, -1.0f),
    XMFLOAT4(Colors::White) }),
    Vertex({ XMFLOAT3(-1.0f, +1.0f, -1.0f),
    XMFLOAT4(Colors::Black) }),
    Vertex({ XMFLOAT3(+1.0f, +1.0f, -1.0f),
    XMFLOAT4(Colors::Red) }),
    Vertex({ XMFLOAT3(+1.0f, -1.0f, -1.0f),
    XMFLOAT4(Colors::Green) }),
    Vertex({ XMFLOAT3(-1.0f, -1.0f, +1.0f),
    XMFLOAT4(Colors::Blue) }),
    Vertex({ XMFLOAT3(-1.0f, +1.0f, +1.0f),
    XMFLOAT4(Colors::Yellow) }),
    Vertex({ XMFLOAT3(+1.0f, +1.0f, +1.0f),
    XMFLOAT4(Colors::Cyan) }),
    Vertex({ XMFLOAT3(+1.0f, -1.0f, +1.0f),
    XMFLOAT4(Colors::Magenta) })
    };
    std::array<std::uint16_t, 36> indices =
    {
        // front face
        0, 1, 2,
        0, 2, 3,
        // back face
        4, 6, 5,
        4, 7, 6,
        // left face
        4, 5, 1,
        4, 1, 0,
        // right face
        3, 2, 6,
        3, 6, 7,
        // top face
        1, 5, 6,
        1, 6, 2,
        // bottom face
        4, 0, 3,
        4, 3, 7
    };
    const UINT vbByteSize = (UINT)vertices.size() *
        sizeof(Vertex);
    const UINT ibByteSize = (UINT)indices.size() *
        sizeof(std::uint16_t);
    mBoxGeo = std::make_unique<MeshGeometry>();
    mBoxGeo->Name = "boxGeo";
    ThrowIfFailed(D3DCreateBlob(vbByteSize, &mBoxGeo -> VertexBufferCPU));
    CopyMemory(mBoxGeo->VertexBufferCPU -> GetBufferPointer(), vertices.data(), vbByteSize);
    ThrowIfFailed(D3DCreateBlob(ibByteSize, &mBoxGeo -> IndexBufferCPU));
    CopyMemory(mBoxGeo->IndexBufferCPU -> GetBufferPointer(), indices.data(), ibByteSize);
    mBoxGeo->VertexBufferGPU =
        CreateDefaultBuffer(vbByteSize, vertices.data(), mBoxGeo->VertexBufferUploader);
    mBoxGeo->IndexBufferGPU =
        CreateDefaultBuffer(ibByteSize, indices.data(), mBoxGeo->IndexBufferUploader);
    mBoxGeo->VertexByteStride = sizeof(Vertex);
    mBoxGeo->VertexBufferByteSize = vbByteSize;
    mBoxGeo->IndexFormat = DXGI_FORMAT_R16_UINT;
    mBoxGeo->IndexBufferByteSize = ibByteSize;
    SubmeshGeometry submesh;
    submesh.IndexCount = (UINT)indices.size();
    submesh.StartIndexLocation = 0;
    submesh.BaseVertexLocation = 0;
    mBoxGeo->DrawArgs["box"] = submesh;
}

void QDirect3D12Widget::BuildPSO()
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc;
    ZeroMemory(&psoDesc,
        sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
    psoDesc.InputLayout = { mpShader->inputLayoutDesc.data(),
    (UINT)mpShader->inputLayoutDesc.size() };
    psoDesc.pRootSignature = mRootSignature.Get();
    psoDesc.VS =
    {
    reinterpret_cast<BYTE*>(mpShader->vsBytecode -> GetBufferPointer()),
    mpShader->vsBytecode->GetBufferSize()
    };
    psoDesc.PS =
    {
    reinterpret_cast<BYTE*>(mpShader->psBytecode -> GetBufferPointer()),
    mpShader->psBytecode->GetBufferSize()
    };
    psoDesc.RasterizerState =
        CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.BlendState =
        CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState =
        CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType =
        D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = mBackBufferFormat;
    psoDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;
    psoDesc.SampleDesc.Quality = m4xMsaaState ?
        (m4xMsaaQuality - 1) : 0;
    psoDesc.DSVFormat = mDepthStencilFormat;
    ThrowIfFailed(md3dDevice -> CreateGraphicsPipelineState(&psoDesc,
    IID_PPV_ARGS(&mPSO)));
}

bool QDirect3D12Widget::init()
{
    InitDirect3D();

    mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr);

    BuildDescriptorHeaps();
    BuildConstantBuffers();
    BuildRootSignature();
    BuildShadersAndInputLayout();
    BuildBoxGeometry();
    BuildPSO();
    //CreateBuffer();

    ThrowIfFailed(mCommandList->Close());
    ID3D12CommandList* cmdLists[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmdLists), cmdLists);

    // Wait until initialization is complete.
    FlushCmdQueue();

    // Start FrameLoop
    connect(&m_qTimer, &QTimer::timeout, this, &QDirect3D12Widget::onFrame);
    // Start Timer
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
    Update();
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

        m_Fps = fps;
        m_TotalTime = timer.TotalTime();

        //为计算下一组帧数值而重置
        frameCnt = 0;
        timeElapsed += 1.0f;
    }
}

//实例化顶点结构体并填充
std::array<Vertex, 8> vertices =
{
    Vertex({ XMFLOAT3(-1.0f, -1.0f, -1.0f), XMFLOAT4(Colors::White) }),
    Vertex({ XMFLOAT3(-1.0f, +1.0f, -1.0f), XMFLOAT4(Colors::Black) }),
    Vertex({ XMFLOAT3(+1.0f, +1.0f, -1.0f), XMFLOAT4(Colors::Red) }),
    Vertex({ XMFLOAT3(+1.0f, -1.0f, -1.0f), XMFLOAT4(Colors::Green) }),
    Vertex({ XMFLOAT3(-1.0f, -1.0f, +1.0f), XMFLOAT4(Colors::Blue) }),
    Vertex({ XMFLOAT3(-1.0f, +1.0f, +1.0f), XMFLOAT4(Colors::Yellow) }),
    Vertex({ XMFLOAT3(+1.0f, +1.0f, +1.0f), XMFLOAT4(Colors::Cyan) }),
    Vertex({ XMFLOAT3(+1.0f, -1.0f, +1.0f), XMFLOAT4(Colors::Magenta) })
};

std::array<std::uint16_t, 36> indices =

{
    //前
    0, 1, 2,
    0, 2, 3,

    //后
    4, 6, 5,
    4, 7, 6,

    //左
    4, 5, 1,
    4, 1, 0,

    //右
    3, 2, 6,
    3, 6, 7,

    //上
    1, 5, 6,
    1, 6, 2,

    //下
    4, 0, 3,
    4, 3, 7
};
const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);
const UINT ibByteSize = (UINT)indices.size() * sizeof(std::uint32_t);

/// <summary>
/// 
/// </summary>
void QDirect3D12Widget::Draw()
{
    //首先重置命令分配器cmdAllocator和命令列表cmdList，目的是重置命令和列表，复用相关内存。
    // Reuse the memory associated with command recording.
    // We can only reset when the associated command lists have finished
    // execution on the GPU.
    ThrowIfFailed(mDirectCmdListAlloc->Reset());//重复使用记录命令的相关内存
    // A command list can be reset after it has been added to the
    // command queue via ExecuteCommandList. Reusing the command
    // list reuses memory.
    //ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));//复用命令列表及其内存
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), mPSO.Get()));//复用命令列表及其内存

    // Indicate a state transition on the resource usage.
    //接着我们将后台缓冲资源从呈现状态转换到渲染目标状态（即准备接收图像渲染）。
    UINT& ref_mCurrentBackBuffer = mCurrentBackBuffer;
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(swapChainBuffer[ref_mCurrentBackBuffer].Get(),//转换资源为后台缓冲区资源
        D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));//从呈现到渲染目标转换

    //接下来设置视口和裁剪矩形。
    mCommandList->RSSetViewports(1, &viewPort);
    mCommandList->RSSetScissorRects(1, &scissorRect);

    // Clear the back buffer and depth buffer.
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

    //D3D12_VERTEX_BUFFER_VIEW vbv;
    //vbv.BufferLocation = VertexBufferGPU->GetGPUVirtualAddress();//顶点缓冲区资源虚拟地址
    //vbv.SizeInBytes = vbByteSize;	//顶点缓冲区大小（所有顶点数据大小）
    //vbv.StrideInBytes = sizeof(Vertex);	//每个顶点元素所占用的字节数

    //设置CBV描述符堆
    //ID3D12DescriptorHeap* descriHeaps[] = { mCbvHeap.Get() };//注意这里之所以是数组，是因为还可能包含SRV和UAV，而这里我们只用到了CBV
    //mCommandList->SetDescriptorHeaps(_countof(descriHeaps), descriHeaps);
    ////设置根签名
    //mCommandList->SetGraphicsRootSignature(mpShader->rootSignature.Get());
    ////将图元拓扑类型传入流水线
    //mCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ////设置根描述符表
    //mCommandList->SetGraphicsRootDescriptorTable(0, //根参数的起始索引
    //    mCbvHeap->GetGPUDescriptorHandleForHeapStart());

    ////绘制顶点（通过索引缓冲区绘制）
    //mCommandList->DrawIndexedInstanced(sizeof(indices), //每个实例要绘制的索引数
    //    1,	//实例化个数
    //    0,	//起始索引位置
    //    0,	//子物体起始索引在全局索引中的位置
    //    0);	//实例化的高级技术，暂时设置为0
    // Specify the buffers we are going to render to. 
    //mCommandList->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());
    //然后我们指定将要渲染的缓冲区，即指定RTV和DSV。
    mCommandList->OMSetRenderTargets(1,//待绑定的RTV数量
        &rtvHandle,	//指向RTV数组的指针
        true,	//RTV对象在堆内存中是连续存放的
        &dsvHandle);	//指向DSV的指针

    ID3D12DescriptorHeap* descriptorHeaps[] = { mCbvHeap.Get() }; 
    mCommandList -> SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
    mCommandList -> SetGraphicsRootSignature(mRootSignature.Get());
    mCommandList->IASetVertexBuffers(0, 1, &mBoxGeo -> VertexBufferView());
    mCommandList->IASetIndexBuffer(&mBoxGeo -> IndexBufferView());
    mCommandList -> IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    mCommandList->SetGraphicsRootDescriptorTable(
        0, mCbvHeap -> GetGPUDescriptorHandleForHeapStart());
    mCommandList->DrawIndexedInstanced(
        mBoxGeo->DrawArgs["box"].IndexCount,
        1, 0, 0, 0);

    // Indicate a state transition on the resource usage.
    // 等到渲染完成，我们要将后台缓冲区的状态改成呈现状态，使其之后推到前台缓冲区显示。完了，关闭命令列表，等待传入命令队列。
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(swapChainBuffer[ref_mCurrentBackBuffer].Get(),
        D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));//从渲染目标到呈现
    // 完成命令的记录关闭命令列表
    ThrowIfFailed(mCommandList->Close());

    // Add the command list to the queue for execution.
    //等CPU将命令都准备好后，需要将待执行的命令列表加入GPU的命令队列。使用的是ExecuteCommandLists函数。
    ID3D12CommandList* commandLists[] = { mCommandList.Get() };//声明并定义命令列表数组
    mCommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);//将命令从命令列表传至命令队列

    // swap the back and front buffers
    ThrowIfFailed(m_SwapChain->Present(0, 0));
    ref_mCurrentBackBuffer = (ref_mCurrentBackBuffer + 1) % 2;

    // Wait until frame commands are complete. This waiting is
    // inefficient and is done for simplicity. Later we will show how to
    // organize our rendering code so we do not have to wait per frame.
    FlushCmdQueue();
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

void QDirect3D12Widget::CreateBuffer()
{
    ////【2】创建堆参数
    ////默认堆,上传堆
    //D3D12_HEAP_PROPERTIES defaultHeap;
    //memset(&defaultHeap, 0, sizeof(defaultHeap));
    //defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

    //D3D12_HEAP_PROPERTIES  uploadheap;
    //memset(&uploadheap, 0, sizeof(uploadheap));
    //uploadheap.Type = D3D12_HEAP_TYPE_UPLOAD;

    ////【3】创建顶点缓冲的资源描述
    ////创建VertexBuffer的资源描述
    //D3D12_RESOURCE_DESC DefaultVertexBufferDesc;
    //memset(&DefaultVertexBufferDesc, 0, sizeof(D3D12_RESOURCE_DESC));
    //DefaultVertexBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    //DefaultVertexBufferDesc.Alignment = 0;
    //DefaultVertexBufferDesc.Width = vbByteSize;
    //DefaultVertexBufferDesc.Height = 1;
    //DefaultVertexBufferDesc.DepthOrArraySize = 1;
    //DefaultVertexBufferDesc.MipLevels = 1;
    //DefaultVertexBufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    //DefaultVertexBufferDesc.SampleDesc.Count = 1;
    //DefaultVertexBufferDesc.SampleDesc.Quality = 0;
    //DefaultVertexBufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    //DefaultVertexBufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ////【4】创建VertexBuffer和VertexBufferUpload
    ////为VertexBuffer和VertexBufferUploader创建资源
    //// Create the actual default buffer resource.
    //ThrowIfFailed(md3dDevice->CreateCommittedResource(
    //    &defaultHeap,
    //    D3D12_HEAP_FLAG_NONE,
    //    &DefaultVertexBufferDesc,
    //    D3D12_RESOURCE_STATE_COPY_DEST,
    //    nullptr,
    //    IID_PPV_ARGS(VertexBufferGPU.GetAddressOf())));

    //ThrowIfFailed(md3dDevice->CreateCommittedResource(
    //    &uploadheap,
    //    D3D12_HEAP_FLAG_NONE,
    //    &DefaultVertexBufferDesc,
    //    D3D12_RESOURCE_STATE_GENERIC_READ,
    //    nullptr,
    //    IID_PPV_ARGS(VertexBufferUploader.GetAddressOf())));

    ////【5】获取 VertexBuffer footprint
    ////获取 VertexBuffer footprint
    //D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    //UINT64  vertex_total_bytes = 0;
    //md3dDevice->GetCopyableFootprints(&DefaultVertexBufferDesc, 0, 1, 0, &footprint, nullptr, nullptr, &vertex_total_bytes);
    //VertexBufferUploader->Unmap(0, nullptr);

    ////【6】映射内存地址,并把数据拷贝到VertexBufferUploader里
    //void* ptr_vertex = nullptr;
    //VertexBufferUploader->Map(0, nullptr, &ptr_vertex);
    //memcpy(reinterpret_cast<char*>(ptr_vertex) + footprint.Offset, vertices.data(), vbByteSize);

    ////【7】拷贝，把VertexBufferUploader里的数据拷贝到VertexBufferGPU里
    //mCommandList->CopyBufferRegion(VertexBufferGPU.Get(), 0, VertexBufferUploader.Get(), 0, vertex_total_bytes);

    ////【8】为VertexBufferGPU插入资源屏障，因为一开始是以D3D12_RESOURCE_STATE_COPY_DEST的状态创建的资源，所以拷贝完以后需要给它设置好资源屏障。
    //D3D12_RESOURCE_BARRIER barrier_vertex;
    //memset(&barrier_vertex, 0, sizeof(barrier_vertex));
    //barrier_vertex.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    //barrier_vertex.Transition.pResource = VertexBufferGPU.Get();
    //barrier_vertex.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    //barrier_vertex.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    //barrier_vertex.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
    //mCommandList->ResourceBarrier(1, &barrier_vertex);

    ////【9】创建IndexBuffer
    //// 到这里我们就完成了对VertexBuffer的处理，下面类似，需要对IndexBuffer进行处理，两者过程及其类似，我就不一步一步赘述了。
    ////创建IndexBuffer的资源描述
    //D3D12_RESOURCE_DESC DefaultIndexBufferDesc;
    //memset(&DefaultIndexBufferDesc, 0, sizeof(D3D12_RESOURCE_DESC));
    //DefaultIndexBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    //DefaultIndexBufferDesc.Alignment = 0;
    //DefaultIndexBufferDesc.Width = ibByteSize;
    //DefaultIndexBufferDesc.Height = 1;
    //DefaultIndexBufferDesc.DepthOrArraySize = 1;
    //DefaultIndexBufferDesc.MipLevels = 1;
    //DefaultIndexBufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    //DefaultIndexBufferDesc.SampleDesc.Count = 1;
    //DefaultIndexBufferDesc.SampleDesc.Quality = 0;
    //DefaultIndexBufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    //DefaultIndexBufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ////为IndexBuffer和IndexBufferUploader创建资源
    //ThrowIfFailed(md3dDevice->CreateCommittedResource(
    //    &defaultHeap,
    //    D3D12_HEAP_FLAG_NONE,
    //    &DefaultIndexBufferDesc,
    //    D3D12_RESOURCE_STATE_COPY_DEST,
    //    nullptr,
    //    IID_PPV_ARGS(IndexBufferGPU.GetAddressOf())));

    //ThrowIfFailed(md3dDevice->CreateCommittedResource(
    //    &uploadheap,
    //    D3D12_HEAP_FLAG_NONE,
    //    &DefaultIndexBufferDesc,
    //    D3D12_RESOURCE_STATE_GENERIC_READ,
    //    nullptr,
    //    IID_PPV_ARGS(IndexBufferUploader.GetAddressOf())));


    ////获取 IndexBuffer footprint
    //D3D12_PLACED_SUBRESOURCE_FOOTPRINT indexBufferFootprint;
    //UINT64  index_total_bytes = 0;
    //md3dDevice->GetCopyableFootprints(&DefaultIndexBufferDesc, 0, 1, 0, &indexBufferFootprint, nullptr, nullptr, &index_total_bytes);


    ////映射内存地址,并把数据拷贝到IndexBufferUploader里
    //void* ptr_index = nullptr;
    //IndexBufferUploader->Map(0, nullptr, &ptr_index);
    //memcpy(reinterpret_cast<char*>(ptr_index) + indexBufferFootprint.Offset, indices.data(), ibByteSize);
    //IndexBufferUploader->Unmap(0, nullptr);

    ////拷贝，把IndexBufferUploader里的数据拷贝到IndexBufferGPU里
    //mCommandList->CopyBufferRegion(IndexBufferGPU.Get(), 0, IndexBufferUploader.Get(), 0, index_total_bytes);

    ////为IndexBufferGPU插入资源屏障
    //D3D12_RESOURCE_BARRIER barrier_index;
    //memset(&barrier_index, 0, sizeof(barrier_index));
    //barrier_index.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    //barrier_index.Transition.pResource = IndexBufferGPU.Get();
    //barrier_index.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    //barrier_index.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    //barrier_index.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
    //mCommandList->ResourceBarrier(1, &barrier_index);

    ////【10】最后我们用VertexBufferGPU和IndexBufferGPU创建VerteBufferView和IndexBufferView来渲染。

    ////VertexBufferView
    //vbv.BufferLocation = VertexBufferGPU->GetGPUVirtualAddress();
    //vbv.StrideInBytes = sizeof(Vertex);
    //vbv.SizeInBytes = vbByteSize;

    ////IndexBufferView
    //ibv.BufferLocation = IndexBufferGPU->GetGPUVirtualAddress();
    //ibv.Format = DXGI_FORMAT_R32_UINT;
    //ibv.SizeInBytes = ibByteSize;

    ////设置顶点缓冲区
    //mCommandList->IASetVertexBuffers(0, 1, &vbv);
    //mCommandList->IASetIndexBuffer(&ibv);

    //Shader shader(md3dDevice);
    //mpShader = &shader;


    //ThrowIfFailed(D3DCreateBlob(vbByteSize, &VertexBufferGPU));	//创建顶点数据内存空间
    //ThrowIfFailed(D3DCreateBlob(ibByteSize, &IndexBufferGPU));	//创建索引数据内存空间
    //CopyMemory(vertexBufferCpu->GetBufferPointer(), vertices.data(), vbByteSize);	//将顶点数据拷贝至顶点系统内存中
    //CopyMemory(indexBufferCpu->GetBufferPointer(), indices.data(), ibByteSize);	//将索引数据拷贝至索引系统内存中
    VertexBufferGPU = CreateDefaultBuffer(vbByteSize, &vertices, VertexBufferUploader);
    IndexBufferGPU = CreateDefaultBuffer(ibByteSize, &indices, IndexBufferUploader);
    //IndexBufferGPU = CreateDefaultBuffer(d3dDevice.Get(), cmdList.Get(), ibByteSize, indices.data(), indexBufferUploader);

    //D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    //ZeroMemory(&psoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
    //psoDesc.InputLayout = { shader.inputLayoutDesc.data(), (UINT)shader.inputLayoutDesc.size() };
    //psoDesc.pRootSignature = shader.rootSignature.Get();
    //psoDesc.VS = { reinterpret_cast<BYTE*>(shader.vsBytecode->GetBufferPointer()), shader.vsBytecode->GetBufferSize() };
    //psoDesc.PS = { reinterpret_cast<BYTE*>(shader.psBytecode->GetBufferPointer()), shader.psBytecode->GetBufferSize() };
    //psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    //psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    //psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    //psoDesc.SampleMask = UINT_MAX;	//0xffffffff,全部采样，没有遮罩
    //psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    //psoDesc.NumRenderTargets = 1;
    //psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;	//归一化的无符号整型
    //psoDesc.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
    //psoDesc.SampleDesc.Count = 1;	//不使用4XMSAA
    //psoDesc.SampleDesc.Quality = 0;	////不使用4XMSAA

    //ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&PSO)));

    //ThrowIfFailed(md3dDevice->CreateRootSignature(0,
    //    shader.serializedRootSig->GetBufferPointer(),
    //    shader.serializedRootSig->GetBufferSize(),
    //    IID_PPV_ARGS(&(shader.rootSignature))));

    D3D12_DESCRIPTOR_HEAP_DESC cbvHeapDesc;
    cbvHeapDesc.NumDescriptors = 1;
    cbvHeapDesc.Type =
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvHeapDesc.Flags =
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    cbvHeapDesc.NodeMask = 0;

    ThrowIfFailed(md3dDevice -> CreateDescriptorHeap(&cbvHeapDesc,
            IID_PPV_ARGS(&mCbvHeap)));
}

/// <summary>
/// 
/// </summary>
/// <param name="byteSize"></param>
/// <param name="initData"></param>
/// <param name="uploadBuffer"></param>
ComPtr<ID3D12Resource> QDirect3D12Widget::CreateDefaultBuffer
    (UINT64 byteSize, const void* initData, ComPtr<ID3D12Resource>& uploadBuffer)
{
    //创建默认堆，作为上传堆的数据传输对象
    ComPtr<ID3D12Resource> defaultBuffer;

    // Create the actual default buffer resource.
    ThrowIfFailed(md3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),//创建默认堆类型的堆
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(byteSize),
        D3D12_RESOURCE_STATE_COMMON,//默认堆为最终存储数据的地方，所以暂时初始化为普通状态
        nullptr,
        IID_PPV_ARGS(defaultBuffer.GetAddressOf())));

    // 创建上传堆，作用是：写入CPU内存数据，并传输给默认堆
    // In order to copy CPU memory data into our default buffer, we need
    // to create an intermediate upload heap.
    ThrowIfFailed(md3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), //创建上传堆类型的堆
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(byteSize),//变体的构造函数，传入byteSize，其他均为默认值，简化书写
        D3D12_RESOURCE_STATE_GENERIC_READ,	//上传堆里的资源需要复制给默认堆，所以是可读状态
        nullptr,	//不是深度模板资源，不用指定优化值
        IID_PPV_ARGS(uploadBuffer.GetAddressOf())));

    // Describe the data we want to copy into the default buffer.
    //将数据从CPU内存拷贝到GPU缓存
    D3D12_SUBRESOURCE_DATA subResourceData;
    subResourceData.pData = initData;
    subResourceData.RowPitch = byteSize;
    subResourceData.SlicePitch = subResourceData.RowPitch;

    //将资源从COMMON状态转换到COPY_DEST状态（默认堆此时作为接收数据的目标）
    // Schedule to copy the data to the default buffer resource. 
    // At a high level, the helper function UpdateSubresources 
    // will copy the CPU memory into the intermediate upload heap. 
    // Then, using ID3D12CommandList::CopySubresourceRegion, 
    // the intermediate upload heap data will be copied to mBuffer.
    mCommandList->ResourceBarrier(1,
        &CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST));

    //核心函数UpdateSubresources，将数据从CPU内存拷贝至上传堆，再从上传堆拷贝至默认堆。1是最大的子资源的下标（模板中定义，意为有2个子资源）
    UpdateSubresources<1>(mCommandList.Get(), defaultBuffer.Get(), uploadBuffer.Get(), 0, 0, 1, &subResourceData);

    //再次将资源从COPY_DEST状态转换到GENERIC_READ状态(现在只提供给着色器访问)
    mCommandList->ResourceBarrier(1,
        &CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_GENERIC_READ));

    // Note: uploadBuffer has to be kept alive after the above function 
    // calls because the command list has not been executed yet that 
    // performs the actual copy. 
    // The caller can Release the uploadBuffer after it knows the copy 
    // has been executed.
    return defaultBuffer;
}
/// <summary>
/// 
/// </summary>
void QDirect3D12Widget::tick()
{
    // TODO: Update your scene here. For aesthetics reasons, only do it here if it's an
    // important component, otherwise do it in the MainWindow.
    // m_pCamera->Tick();

    emit ticked();
}

void QDirect3D12Widget::onReset()
{
    // TODO(Gilad): FIXME: this needs to be done in a synchronized manner. Need to look at
    // DirectX-12 samples here: https://github.com/microsoft/DirectX-Graphics-Samples how to
    // properly do this without leaking memory.
    pauseFrames();
    //resizeSwapChain(width(), height());
    continueFrames();
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
int i = 0;

int mLastMousePosx = 0;
int mLastMousePosy = 0;

void QDirect3D12Widget::Update()
{
     //Convert Spherical to Cartesian coordinates. 
    float x = mRadius*sinf(mPhi)*cosf(mTheta); 
    float z = mRadius*sinf(mPhi)*sinf(mTheta); 
    float y = mRadius*cosf(mPhi); 
    // Build the view matrix. 
    
    XMVECTOR pos = XMVectorSet(x, y, z, 1.0f); 
    XMVECTOR target = XMVectorZero(); 
    XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f); 
    XMMATRIX view = XMMatrixLookAtLH(pos, target, up); 
    XMStoreFloat4x4(&mView, view); 
    XMMATRIX world = XMLoadFloat4x4(&mWorld); 
    XMMATRIX proj = XMLoadFloat4x4(&mProj);

    XMMATRIX worldViewProj = world * view * proj; // Update the constant buffer with the latest worldViewProj matrix. ObjectConstants objConstants; XMStoreFloat4x4(&objConstants.WorldViewProj, XMMatrixTranspose(worldViewProj)); mObjectCB->CopyData(0, objConstants);
    // Update the constant buffer with the latest worldViewProj matrix.
    ObjectConstants objConstants;
    XMStoreFloat4x4(&objConstants.WorldViewProj,
        XMMatrixTranspose(worldViewProj));
    mObjectCB->CopyData(0, objConstants);
}

void QDirect3D12Widget::OnMouseMove(QMouseEvent* event)
{
    int x = event->pos().x();
    int y = event->pos().y();

    float dx = XMConvertToRadians(0.25f * static_cast<float> (x - mLastMousePosx));
    float dy = XMConvertToRadians(0.25f * static_cast<float> (y - mLastMousePosy));

    mTheta += dx; 
    mPhi += dy;

    mPhi = MathHelper::Clamp(mPhi, 0.1f, MathHelper::Pi - 0.1f);
    // 
    mLastMousePosx = x;
    mLastMousePosy = y;
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
            OnMouseMove((QMouseEvent*)event);
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
