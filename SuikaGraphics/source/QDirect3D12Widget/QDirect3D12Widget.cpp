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

//���嶥��ṹ��
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
    // Set palette
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

/// <summary>
/// LIFECYCLE :: Release
/// </summary>
void QDirect3D12Widget::Release()
{
    m_bDeviceInitialized = false;
    disconnect(&m_qTimer, &QTimer::timeout, this, &QDirect3D12Widget::onFrame);
    m_qTimer.stop();
}

/// <summary>
/// LIFECYCLE :: Initialization
/// </summary>
void QDirect3D12Widget::Run()
{
    m_qTimer.start(MS_PER_FRAME);
    m_bRenderActive = m_bStarted = true;
}

/// <summary>
/// LIFECYCLE :: Pause
/// </summary>
void QDirect3D12Widget::PauseFrames()
{
    if (!m_qTimer.isActive() || !m_bStarted) return;

    disconnect(&m_qTimer, &QTimer::timeout, this, &QDirect3D12Widget::onFrame);
    m_qTimer.stop();
    m_tGameTimer.Stop();
    m_bRenderActive = false;
}

/// <summary>
/// LIFECYCLE :: Continue
/// </summary>
void QDirect3D12Widget::ContinueFrames()
{
    if (m_qTimer.isActive() || !m_bStarted) return;

    connect(&m_qTimer, &QTimer::timeout, this, &QDirect3D12Widget::onFrame);
    m_qTimer.start(MS_PER_FRAME);
    m_tGameTimer.Start();
    m_bRenderActive = true;
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

#pragma region LIFECYCLE

/// <summary>
/// LIFECYCLE :: Initialization (First Show)
/// </summary>
bool QDirect3D12Widget::Initialize()
{
    onResize();
    InitDirect3D();

    mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr);

    BuildDescriptorHeaps();
    BuildConstantBuffers();
    BuildRootSignature();
    BuildShadersAndInputLayout();
    BuildBoxGeometry();
    BuildPSO();

    ThrowIfFailed(mCommandList->Close());
    ID3D12CommandList* cmdLists[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmdLists), cmdLists);

    // Wait until initialization is complete.
    FlushCmdQueue();

    // Start FrameLoop
    connect(&m_qTimer, &QTimer::timeout, this, &QDirect3D12Widget::onFrame);
    // Start Timer
    m_tGameTimer.Reset();

    return true;
}
/// <summary>
/// LIFECYCLE :: Update (Before Draw)
/// </summary>
void QDirect3D12Widget::Update()
{
    //Convert Spherical to Cartesian coordinates. 
    float x = mRadius * sinf(mPhi) * cosf(mTheta);
    float z = mRadius * sinf(mPhi) * sinf(mTheta);
    float y = mRadius * cosf(mPhi);
    // Build the view matrix. 

    XMVECTOR pos = XMVectorSet(x, y, z, 1.0f);
    XMVECTOR target = XMVectorZero();
    XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
    XMMATRIX view = XMMatrixLookAtLH(pos, target, up);
    XMStoreFloat4x4(&mView, view);
    XMMATRIX world = XMLoadFloat4x4(&mWorld);
    XMMATRIX proj = XMLoadFloat4x4(&mProj);

    //����ͶӰ����
    //XMMATRIX proj = XMMatrixPerspectiveFovLH(0.25f * 3.1416f, 1280.0f / 720.0f, 1.0f, 1000.0f);

    XMMATRIX worldViewProj = world * view * proj; // Update the constant buffer with the latest worldViewProj matrix. ObjectConstants objConstants; XMStoreFloat4x4(&objConstants.WorldViewProj, XMMatrixTranspose(worldViewProj)); mObjectCB->CopyData(0, objConstants);
    // Update the constant buffer with the latest worldViewProj matrix.
    ObjectConstants objConstants;
    XMStoreFloat4x4(&objConstants.WorldViewProj, XMMatrixTranspose(worldViewProj));
    mObjectCB->CopyData(0, objConstants);
}
/// <summary>
/// LIFECYCLE :: Draw Stuff
/// </summary>
void QDirect3D12Widget::Draw()
{
    //�����������������cmdAllocator�������б�cmdList��Ŀ��������������б���������ڴ档
    // Reuse the memory associated with command recording.
    // We can only reset when the associated command lists have finished
    // execution on the GPU.
    ThrowIfFailed(mDirectCmdListAlloc->Reset());//�ظ�ʹ�ü�¼���������ڴ�
    // A command list can be reset after it has been added to the
    // command queue via ExecuteCommandList. Reusing the command
    // list reuses memory.
    //ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));//���������б����ڴ�
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), mPSO.Get()));//���������б����ڴ�

    // Indicate a state transition on the resource usage.
    //�������ǽ���̨������Դ�ӳ���״̬ת������ȾĿ��״̬����׼������ͼ����Ⱦ����
    UINT& ref_mCurrentBackBuffer = mCurrentBackBuffer;
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(swapChainBuffer[ref_mCurrentBackBuffer].Get(),//ת����ԴΪ��̨��������Դ
        D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));//�ӳ��ֵ���ȾĿ��ת��

    //�����������ӿںͲü����Ρ�
    mCommandList->RSSetViewports(1, &viewPort);
    mCommandList->RSSetScissorRects(1, &scissorRect);

    // Clear the back buffer and depth buffer.
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

    // Specify the buffers we are going to render to. 
    //mCommandList->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());
    //Ȼ������ָ����Ҫ��Ⱦ�Ļ���������ָ��RTV��DSV��
    mCommandList->OMSetRenderTargets(1,//���󶨵�RTV����
        &rtvHandle,	//ָ��RTV�����ָ��
        true,	//RTV�����ڶ��ڴ�����������ŵ�
        &dsvHandle);	//ָ��DSV��ָ��

    ID3D12DescriptorHeap* descriptorHeaps[] = { mCbvHeap.Get() };
    mCommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
    mCommandList->SetGraphicsRootSignature(mRootSignature.Get());
    mCommandList->IASetVertexBuffers(0, 1, &mBoxGeo->VertexBufferView());
    mCommandList->IASetIndexBuffer(&mBoxGeo->IndexBufferView());
    mCommandList->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    mCommandList->SetGraphicsRootDescriptorTable(
        0, mCbvHeap->GetGPUDescriptorHandleForHeapStart());
    mCommandList->DrawIndexedInstanced(
        mBoxGeo->DrawArgs["box"].IndexCount,
        1, 0, 0, 0);

    // Indicate a state transition on the resource usage.
    // �ȵ���Ⱦ��ɣ�����Ҫ����̨��������״̬�ĳɳ���״̬��ʹ��֮���Ƶ�ǰ̨��������ʾ�����ˣ��ر������б��ȴ�����������С�
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(swapChainBuffer[ref_mCurrentBackBuffer].Get(),
        D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));//����ȾĿ�굽����
    // �������ļ�¼�ر������б�
    ThrowIfFailed(mCommandList->Close());

    // Add the command list to the queue for execution.
    //��CPU�����׼���ú���Ҫ����ִ�е������б����GPU��������С�ʹ�õ���ExecuteCommandLists������
    ID3D12CommandList* commandLists[] = { mCommandList.Get() };//���������������б�����
    mCommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);//������������б����������

    // swap the back and front buffers
    ThrowIfFailed(m_SwapChain->Present(0, 0));
    ref_mCurrentBackBuffer = (ref_mCurrentBackBuffer + 1) % 2;

    // Wait until frame commands are complete. This waiting is
    // inefficient and is done for simplicity. Later we will show how to
    // organize our rendering code so we do not have to wait per frame.
    FlushCmdQueue();
}

#pragma endregion

#pragma region QtSlot

/// <summary>
/// QT Slot :: On Frame
/// </summary>
void QDirect3D12Widget::onFrame()
{
    // Send ticked signal
    if (m_bRenderActive) emit ticked();

    m_tGameTimer.Tick();

    CalculateFrameState();
    Update();
    Draw();
}

/// <summary>
/// QT Slot :: On Resize
/// </summary>
void QDirect3D12Widget::onResize()
{
    // TODO(Gilad): FIXME: this needs to be done in a synchronized manner. Need to look at
    // DirectX-12 samples here: https://github.com/microsoft/DirectX-Graphics-Samples how to
    // properly do this without leaking memory.
    PauseFrames();
    // The window resized, so update the aspect ratio and recompute the
        // projection matrix.
    XMMATRIX P =
        XMMatrixPerspectiveFovLH(0.25f * MathHelper::Pi, (float)(width()) / (height()), 1.0f, 1000.0f);
    XMStoreFloat4x4(&mProj, P);

    //resizeSwapChain(width(), height());
    ContinueFrames();
}
#pragma endregion

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
    if (m_tGameTimer.TotalTime() - timeElapsed >= 1.0f)	//һ��>=0��˵���պù�һ��
    {
        float fps = (float)frameCnt;//ÿ�����֡
        float mspf = 1000.0f / fps;	//ÿ֡���ٺ���

        m_Fps = fps;
        m_TotalTime = m_tGameTimer.TotalTime();

        //Ϊ������һ��֡��ֵ������
        frameCnt = 0;
        timeElapsed += 1.0f;
    }
}

//ʵ��������ṹ�岢���
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
    //ǰ
    0, 1, 2,
    0, 2, 3,

    //��
    4, 6, 5,
    4, 7, 6,

    //��
    4, 5, 1,
    4, 1, 0,

    //��
    3, 2, 6,
    3, 6, 7,

    //��
    1, 5, 6,
    1, 6, 2,

    //��
    4, 0, 3,
    4, 3, 7
};
const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);
const UINT ibByteSize = (UINT)indices.size() * sizeof(std::uint32_t);

/// <summary>
/// 
/// </summary>
/// <param name="byteSize"></param>
/// <param name="initData"></param>
/// <param name="uploadBuffer"></param>
ComPtr<ID3D12Resource> QDirect3D12Widget::CreateDefaultBuffer
    (UINT64 byteSize, const void* initData, ComPtr<ID3D12Resource>& uploadBuffer)
{
    //����Ĭ�϶ѣ���Ϊ�ϴ��ѵ����ݴ������
    ComPtr<ID3D12Resource> defaultBuffer;

    // Create the actual default buffer resource.
    ThrowIfFailed(md3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),//����Ĭ�϶����͵Ķ�
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(byteSize),
        D3D12_RESOURCE_STATE_COMMON,//Ĭ�϶�Ϊ���մ洢���ݵĵط���������ʱ��ʼ��Ϊ��ͨ״̬
        nullptr,
        IID_PPV_ARGS(defaultBuffer.GetAddressOf())));

    // �����ϴ��ѣ������ǣ�д��CPU�ڴ����ݣ��������Ĭ�϶�
    // In order to copy CPU memory data into our default buffer, we need
    // to create an intermediate upload heap.
    ThrowIfFailed(md3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), //�����ϴ������͵Ķ�
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(byteSize),//����Ĺ��캯��������byteSize��������ΪĬ��ֵ������д
        D3D12_RESOURCE_STATE_GENERIC_READ,	//�ϴ��������Դ��Ҫ���Ƹ�Ĭ�϶ѣ������ǿɶ�״̬
        nullptr,	//�������ģ����Դ������ָ���Ż�ֵ
        IID_PPV_ARGS(uploadBuffer.GetAddressOf())));

    // Describe the data we want to copy into the default buffer.
    //�����ݴ�CPU�ڴ濽����GPU����
    D3D12_SUBRESOURCE_DATA subResourceData;
    subResourceData.pData = initData;
    subResourceData.RowPitch = byteSize;
    subResourceData.SlicePitch = subResourceData.RowPitch;

    //����Դ��COMMON״̬ת����COPY_DEST״̬��Ĭ�϶Ѵ�ʱ��Ϊ�������ݵ�Ŀ�꣩
    // Schedule to copy the data to the default buffer resource. 
    // At a high level, the helper function UpdateSubresources 
    // will copy the CPU memory into the intermediate upload heap. 
    // Then, using ID3D12CommandList::CopySubresourceRegion, 
    // the intermediate upload heap data will be copied to mBuffer.
    mCommandList->ResourceBarrier(1,
        &CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST));

    //���ĺ���UpdateSubresources�������ݴ�CPU�ڴ濽�����ϴ��ѣ��ٴ��ϴ��ѿ�����Ĭ�϶ѡ�1����������Դ���±꣨ģ���ж��壬��Ϊ��2������Դ��
    UpdateSubresources<1>(mCommandList.Get(), defaultBuffer.Get(), uploadBuffer.Get(), 0, 0, 1, &subResourceData);

    //�ٴν���Դ��COPY_DEST״̬ת����GENERIC_READ״̬(����ֻ�ṩ����ɫ������)
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

int i = 0;

int mLastMousePosx = 0;
int mLastMousePosy = 0;

#pragma region InputCallback

void QDirect3D12Widget::OnMouseMove(QMouseEvent* event)
{
    int x = event->pos().x();
    int y = event->pos().y();

    float dx = XMConvertToRadians(0.25f * static_cast<float> (x - mLastMousePosx));
    float dy = XMConvertToRadians(0.25f * static_cast<float> (y - mLastMousePosy));

    mTheta -= dx;
    mPhi -= dy;

    mPhi = MathHelper::Clamp(mPhi, 0.1f, MathHelper::Pi - 0.1f);
    // 
    mLastMousePosx = x;
    mLastMousePosy = y;
}

void QDirect3D12Widget::OnMousePressed(QMouseEvent* event)
{
    int x = event->pos().x();
    int y = event->pos().y();

    mLastMousePosx = x;
    mLastMousePosy = y;
}

void QDirect3D12Widget::OnMouseReleased(QMouseEvent* event)
{

}
#pragma endregion

#pragma region D3DInitialize
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

#pragma region QtOverrideEvent

/// <summary>
/// QT OVERRIDE EVENT :: Event
/// -------------------------------------
/// Deal with all kinds of input events
/// -------------------------------------
/// </summary>
bool QDirect3D12Widget::event(QEvent* event)
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
            QWidget* nativeParent = this;
            while (true)
            {
                if (nativeParent->isWindow()) break;

                QWidget* parent = nativeParent->nativeParentWidget();
                if (!parent) break;

                nativeParent = parent;
            }

            if (nativeParent && nativeParent != this &&
                ::GetFocus() == reinterpret_cast<HWND>(nativeParent->winId()))
                ::SetFocus(m_hWnd);
        }
        break;
    case QEvent::KeyPress:
        if (!m_bRenderActive) break;
        emit keyPressed((QKeyEvent*)event);
        break;
    case QEvent::MouseMove:
        if (!m_bRenderActive) break;
        OnMouseMove((QMouseEvent*)event);
        emit mouseMoved((QMouseEvent*)event);
        break;
    case QEvent::MouseButtonPress:
        if (!m_bRenderActive) break;
        OnMousePressed((QMouseEvent*)event);
        emit mouseClicked((QMouseEvent*)event);
        break;
    case QEvent::MouseButtonRelease:
        if (!m_bRenderActive) break;
        OnMouseReleased((QMouseEvent*)event);
        emit mouseReleased((QMouseEvent*)event);
        break;
    }

    return QWidget::event(event);
}

/// <summary>
/// QT OVERRIDE EVENT :: ShowEvent
/// --------------------------------------------------------------------------
/// Actually kinds of initialization
/// Internal show events are delivered just before the widget becomes visible.
/// --------------------------------------------------------------------------
/// </summary>
void QDirect3D12Widget::showEvent(QShowEvent* event)
{
    if (!m_bDeviceInitialized)
    {
        m_bDeviceInitialized = Initialize();
        emit deviceInitialized(m_bDeviceInitialized);
    }

    QWidget::showEvent(event);
}

/// <summary>
/// QT OVERRIDE EVENT :: PaintEngine
/// -------------------------------------
/// provides an abstract definition of how 
/// QPainter draws to a given device on a given platform
/// -------------------------------------
/// </summary>
QPaintEngine* QDirect3D12Widget::paintEngine() const
{
    return Q_NULLPTR;
}

/// <summary>
/// QT OVERRIDE EVENT :: PaintEvent
/// -------------------------------------
/// -------------------------------------
/// </summary>
void QDirect3D12Widget::paintEvent(QPaintEvent* event) {}

/// <summary>
/// QT OVERRIDE EVENT :: ResizeEvent
/// -------------------------------------
/// Help with parent class to resize, 
/// also call onResize in the widget.
/// -------------------------------------
/// </summary>
void QDirect3D12Widget::resizeEvent(QResizeEvent* event)
{
    if (m_bDeviceInitialized)
    {
        //Debug Change
        onResize();
        emit widgetResized();
    }

    QWidget::resizeEvent(event);
}

/// <summary>
/// QT OVERRIDE EVENT :: WheelEvent
/// -------------------------------------
/// Deal with wheel input event
/// -------------------------------------
/// </summary>
void QDirect3D12Widget::wheelEvent(QWheelEvent* event)
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
#pragma endregion


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
