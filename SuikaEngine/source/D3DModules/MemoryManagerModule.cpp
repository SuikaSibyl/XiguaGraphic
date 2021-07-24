#include <Precompiled.h>
#include <MemoryManagerModule.h>
#include <SynchronizationModule.h>
#include <ScreenGrab.h>

//======================================================================================
//======================================================================================
// RenderTargetSubmodule
//======================================================================================
//======================================================================================
D3DModules::RTDSSubmodule::RTDSSubmodule(ID3D12Device* device, MemoryManagerModule* mmmodule) :m_d3dDevice(device)
{
    MMModule = mmmodule;

    // Calc sizes
    m_dsvDescriptorSize = m_d3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    m_rtvDescriptorSize = m_d3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
}

void D3DModules::RTDSSubmodule::PrintWirtableTexture(std::string name, const wchar_t* file, ID3D12CommandQueue* ptrCmdQueue)
{
    if (mRTWritableTextures.find(name) == mRTWritableTextures.end())
    {
        Debug::LogError("Print Failed: No such writable texture!");
        return;
    }

    mRTWritableTextures[name]->CaptureTexture(file, GUID_ContainerFormatBmp, ptrCmdQueue);
}

void D3DModules::RTDSSubmodule::RenderToScreen()
{
	// Set the Viewport & ScissorRects
    SetViewportsScissor();
    CleanPreviousRenderTarget();

    // Set the state of new Render Target
    D3D12_RESOURCE_STATES prev_state;
    ID3D12Resource* resource_addr;
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;

    prevRTName = "NONE";
    prev_state = D3D12_RESOURCE_STATE_PRESENT;
    resource_addr = m_SwapChainBuffer[m_CurrentBackBuffer].Get();
    rtvHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), m_CurrentBackBuffer, m_rtvDescriptorSize);

    // Indicate a state transition on the resource usage.
    //�������ǽ���̨������Դ�ӳ���״̬ת������ȾĿ��״̬����׼������ͼ����Ⱦ����
    ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(resource_addr,//ת����ԴΪ��̨��������Դ
        screenRTState, D3D12_RESOURCE_STATE_RENDER_TARGET));//�ӳ��ֵ���ȾĿ��ת��
    screenRTState = D3D12_RESOURCE_STATE_RENDER_TARGET;
    // Clear the back buffer and depth buffer.
    //Ȼ�������̨����������Ȼ�����������ֵ���������Ȼ�ö������������������ַ������ͨ��ClearRenderTargetView������ClearDepthStencilView����������͸�ֵ���������ǽ�RT��Դ����ɫ��ֵΪDarkRed�����죩��
    float dark[4] = { 0.117,0.117,0.117,1 };
    ptr_CommandList->ClearRenderTargetView(rtvHandle, dark, 0, nullptr);//���RT����ɫΪ���죬���Ҳ����òü�����

    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    ptr_CommandList->ClearDepthStencilView(dsvHandle,	//DSV���������
        D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
        1.0f,	//Ĭ�����ֵ
        0,	//Ĭ��ģ��ֵ
        0,	//�ü���������
        nullptr);	//�ü�����ָ��

    // Specify the buffers we are going to render to. 
    //Ȼ������ָ����Ҫ��Ⱦ�Ļ���������ָ��RTV��DSV��
    ptr_CommandList->OMSetRenderTargets(1,//���󶨵�RTV����
        &rtvHandle,	//ָ��RTV�����ָ��
        true,	//RTV�����ڶ��ڴ�����������ŵ�
        &dsvHandle);	//ָ��DSV��ָ��
}

void D3DModules::RTDSSubmodule::RenderToTexture(std::string name, std::string dsname)
{
    CleanPreviousRenderTarget();

    // Set the state of new Render Target
    D3D12_RESOURCE_STATES prev_state;
    ID3D12Resource* resource_addr;
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;

    // Set RTV
    // ------------------------------------------
    prevRTName = name;
    prev_state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    rtvHandle = mRTWritableTextures[name]->Rtv();
    resource_addr = mRTWritableTextures[name]->Resource();
    //����Viewports��ScissorRects
    ptr_CommandList->RSSetViewports(1, &mRTWritableTextures[name]->Viewport());
    ptr_CommandList->RSSetScissorRects(1, &mRTWritableTextures[name]->ScissorRect());
    // Indicate a state transition on the resource usage.
    //�������ǽ���̨������Դ�ӳ���״̬ת������ȾĿ��״̬����׼������ͼ����Ⱦ����
    mRTWritableTextures[name]->ChangeResourceState(ptr_CommandList, D3D12_RESOURCE_STATE_RENDER_TARGET);
    //ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(resource_addr,//ת����ԴΪ��̨��������Դ
    //    prev_state, D3D12_RESOURCE_STATE_RENDER_TARGET));//�ӳ��ֵ���ȾĿ��ת��
    // Clear the back buffer and depth buffer.
    //Ȼ�������̨����������Ȼ�����������ֵ���������Ȼ�ö������������������ַ������ͨ��ClearRenderTargetView������ClearDepthStencilView����������͸�ֵ���������ǽ�RT��Դ����ɫ��ֵΪDarkRed�����죩��
    float dark[4] = { 0.117,0.117,0.117,1 };
    ptr_CommandList->ClearRenderTargetView(rtvHandle, dark, 0, nullptr);//���RT����ɫΪ���죬���Ҳ����òü�����

    // Set DSV
    // ------------------------------------------
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    if (dsname != "DEFAULT")
    {
        dsvHandle = mDSWritableTextures[dsname]->Dsv();
    }
    ptr_CommandList->ClearDepthStencilView(dsvHandle,	//DSV���������
        D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
        1.0f,	//Ĭ�����ֵ
        0,	//Ĭ��ģ��ֵ
        0,	//�ü���������
        nullptr);	//�ü�����ָ��

    // Specify the buffers we are going to render to. 
    //Ȼ������ָ����Ҫ��Ⱦ�Ļ���������ָ��RTV��DSV��
    ptr_CommandList->OMSetRenderTargets(1,//���󶨵�RTV����
        &rtvHandle,	//ָ��RTV�����ָ��
        true,	//RTV�����ڶ��ڴ�����������ŵ�
        &dsvHandle);	//ָ��DSV��ָ��
}

void D3DModules::RTDSSubmodule::RenderToTexture(std::string name, std::string name2, std::string dsname)
{
    //// Set RTV
    //// ------------------------------------------
    //resource_addr = mRTWritableTextures[name]->Resource();
    ////����Viewports��ScissorRects
    //ptr_CommandList->RSSetViewports(1, &mRTWritableTextures[name]->Viewport());
    //ptr_CommandList->RSSetScissorRects(1, &mRTWritableTextures[name]->ScissorRect());
    //// Indicate a state transition on the resource usage.
    ////�������ǽ���̨������Դ�ӳ���״̬ת������ȾĿ��״̬����׼������ͼ����Ⱦ����
    //mRTWritableTextures[name]->ChangeResourceState(ptr_CommandList, D3D12_RESOURCE_STATE_RENDER_TARGET);
    //mRTWritableTextures[name2]->ChangeResourceState(ptr_CommandList, D3D12_RESOURCE_STATE_RENDER_TARGET);
    ////ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(resource_addr,//ת����ԴΪ��̨��������Դ
    ////    prev_state, D3D12_RESOURCE_STATE_RENDER_TARGET));//�ӳ��ֵ���ȾĿ��ת��
    //// Clear the back buffer and depth buffer.
    ////Ȼ�������̨����������Ȼ�����������ֵ���������Ȼ�ö������������������ַ������ͨ��ClearRenderTargetView������ClearDepthStencilView����������͸�ֵ���������ǽ�RT��Դ����ɫ��ֵΪDarkRed�����죩��
    //float dark[4] = { 0.117,0.117,0.117,1 };
    //ptr_CommandList->ClearRenderTargetView(rtvHandle, dark, 0, nullptr);//���RT����ɫΪ���죬���Ҳ����òü�����
    ////ptr_CommandList->ClearRenderTargetView(rtv2Handle, dark, 0, nullptr);//���RT����ɫΪ���죬���Ҳ����òü�����

    //// Set DSV
    //// ------------------------------------------
    //D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    //if (dsname != "DEFAULT")
    //{
    //    dsvHandle = mDSWritableTextures[dsname]->Dsv();
    //}
    //ptr_CommandList->ClearDepthStencilView(dsvHandle,	//DSV���������
    //    D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
    //    1.0f,	//Ĭ�����ֵ
    //    0,	//Ĭ��ģ��ֵ
    //    0,	//�ü���������
    //    nullptr);	//�ü�����ָ��

    //D3D12_CPU_DESCRIPTOR_HANDLE* handle_array = new D3D12_CPU_DESCRIPTOR_HANDLE();
    //handle_array[0] = rtvHandle;
    //handle_array[1] = rtv2Handle;
    //// Specify the buffers we are going to render to. 
    ////Ȼ������ָ����Ҫ��Ⱦ�Ļ���������ָ��RTV��DSV��
    //ptr_CommandList->OMSetRenderTargets(1,//���󶨵�RTV����
    //    &rtvHandle,	//ָ��RTV�����ָ��
    //    true,	//RTV�����ڶ��ڴ�����������ŵ�
    //    &dsvHandle);	//ָ��DSV��ָ��


    CleanPreviousRenderTarget();

    // Set the state of new Render Target
    D3D12_RESOURCE_STATES prev_state;
    ID3D12Resource* resource_addr;
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv2Handle;

    // Set RTV
    // ------------------------------------------
    prevRTName = name;
    prevRT2Name = name2;
    prev_state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    rtvHandle = mRTWritableTextures[name]->Rtv();
    rtv2Handle = mRTWritableTextures[name2]->Rtv();
    //����Viewports��ScissorRects
    ptr_CommandList->RSSetViewports(1, &mRTWritableTextures[name]->Viewport());
    ptr_CommandList->RSSetScissorRects(1, &mRTWritableTextures[name]->ScissorRect());
    // Indicate a state transition on the resource usage.
    //�������ǽ���̨������Դ�ӳ���״̬ת������ȾĿ��״̬����׼������ͼ����Ⱦ����
    mRTWritableTextures[name]->ChangeResourceState(ptr_CommandList, D3D12_RESOURCE_STATE_RENDER_TARGET);
    mRTWritableTextures[name2]->ChangeResourceState(ptr_CommandList, D3D12_RESOURCE_STATE_RENDER_TARGET);
    //ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(resource_addr,//ת����ԴΪ��̨��������Դ
    //    prev_state, D3D12_RESOURCE_STATE_RENDER_TARGET));//�ӳ��ֵ���ȾĿ��ת��
    // Clear the back buffer and depth buffer.
    //Ȼ�������̨����������Ȼ�����������ֵ���������Ȼ�ö������������������ַ������ͨ��ClearRenderTargetView������ClearDepthStencilView����������͸�ֵ���������ǽ�RT��Դ����ɫ��ֵΪDarkRed�����죩��
    float dark[4] = { 0.117,0.117,0.117,1 };
    ptr_CommandList->ClearRenderTargetView(rtvHandle, dark, 0, nullptr);//���RT����ɫΪ���죬���Ҳ����òü�����
    ptr_CommandList->ClearRenderTargetView(rtv2Handle, dark, 0, nullptr);//���RT����ɫΪ���죬���Ҳ����òü�����

    // Set DSV
    // ------------------------------------------
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    if (dsname != "DEFAULT")
    {
        dsvHandle = mDSWritableTextures[dsname]->Dsv();
    }
    ptr_CommandList->ClearDepthStencilView(dsvHandle,	//DSV���������
        D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
        1.0f,	//Ĭ�����ֵ
        0,	//Ĭ��ģ��ֵ
        0,	//�ü���������
        nullptr);	//�ü�����ָ��

    D3D12_CPU_DESCRIPTOR_HANDLE* handle_array = new D3D12_CPU_DESCRIPTOR_HANDLE[2];
    handle_array[0] = rtvHandle;
    handle_array[1] = rtv2Handle;

    // Specify the buffers we are going to render to. 
    //Ȼ������ָ����Ҫ��Ⱦ�Ļ���������ָ��RTV��DSV��
    ptr_CommandList->OMSetRenderTargets(2,//���󶨵�RTV����
        handle_array,	//ָ��RTV�����ָ��
        false,	//RTV�����ڶ��ڴ�����������ŵ�
        &dsvHandle);	//ָ��DSV��ָ��
}
void D3DModules::RTDSSubmodule::CleanPreviousRenderTarget()
{
    //// �ȵ���Ⱦ��ɣ�����Ҫ����̨��������״̬�ĳɳ���״̬��ʹ��֮���Ƶ�ǰ̨��������ʾ�����ˣ��ر������б��ȴ�����������С�
    if (prevRTName != "NONE")
        mRTWritableTextures[prevRTName]->ChangeResourceState(
            ptr_CommandList,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    if (prevRT2Name != "NONE")
        mRTWritableTextures[prevRT2Name]->ChangeResourceState(
            ptr_CommandList,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    if (prevRT3Name != "NONE")
        mRTWritableTextures[prevRT3Name]->ChangeResourceState(
            ptr_CommandList,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    prevRTName = "NONE";
    prevRT2Name = "NONE";
    prevRT3Name = "NONE";
 }

void D3DModules::RTDSSubmodule::RenderTextureToScreen(std::string name)
{
    ID3D12Resource* screen_addr = m_SwapChainBuffer[m_CurrentBackBuffer].Get();
    ID3D12Resource* resource_addr = mRTWritableTextures[name]->Resource();

    mRTWritableTextures[name]->ChangeResourceState(
        ptr_CommandList,
        D3D12_RESOURCE_STATE_COPY_SOURCE);

    //����̨������Դת�ɡ�����Ŀ�ꡱ
    ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        screen_addr,
        screenRTState,
        D3D12_RESOURCE_STATE_COPY_DEST));
    screenRTState = D3D12_RESOURCE_STATE_COPY_DEST;

    //��ģ����������������������̨������
    ptr_CommandList->CopyResource(screen_addr, resource_addr);

    CleanPreviousRenderTarget();
}

void D3DModules::RTDSSubmodule::UnorderedAccessTextureToScreen(std::string name)
{
    ID3D12Resource* screen_addr = m_SwapChainBuffer[m_CurrentBackBuffer].Get();
    ID3D12Resource* resource_addr = mUAWritableTextures[name]->Resource();

    mUAWritableTextures[name]->ChangeResourceState(
        ptr_CommandList,
        D3D12_RESOURCE_STATE_COPY_SOURCE);

    //����̨������Դת�ɡ�����Ŀ�ꡱ
    ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        screen_addr,
        screenRTState,
        D3D12_RESOURCE_STATE_COPY_DEST));
    screenRTState = D3D12_RESOURCE_STATE_COPY_DEST;

    //��ģ����������������������̨������
    ptr_CommandList->CopyResource(screen_addr, resource_addr);

    CleanPreviousRenderTarget();
}

WritableTexture* D3DModules::RTDSSubmodule::CreateWritableTexture(std::string name, UINT width, UINT height, WritableTexture::WritableType type)
{
    if (width == -1)
        width = this->width;
    if (height == -1)
        height = this->height;

    switch (type)
    {
    case WritableTexture::RenderTarget:
        // Create Writable Texture
        mRTWritableTextures[name] = std::make_unique<WritableTexture>(m_d3dDevice, width, height, type);
        return mRTWritableTextures[name].get();
        break;
    case WritableTexture::DepthStencil:
        // Create Writable Texture
        mDSWritableTextures[name] = std::make_unique<WritableTexture>(m_d3dDevice, width, height, type);
        return mDSWritableTextures[name].get();
        break;
    case WritableTexture::UnorderedAccess:
        mUAWritableTextures[name] = std::make_unique<WritableTexture>(m_d3dDevice, width, height, type);
        return mUAWritableTextures[name].get();
        break;
    case WritableTexture::CudaShared:
        mUAWritableTextures[name] = std::make_unique<WritableTexture>(m_d3dDevice, width, height, type);
        return mUAWritableTextures[name].get();
        break;
    default:
        break;
    }
}

void D3DModules::RTDSSubmodule::CreateRTDSHeap(float width, float height, IDXGISwapChain* swapChain)
{
    m_SwapChain = swapChain;
    this->width = width;
    this->height = height;
    CreateRTVHeap();
    CreateDSVHeap();
}

void D3DModules::RTDSSubmodule::CreateRTVHeap()
{
    // Initialize Descriptor Heap
    // -----------------------------------------------
    D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc;
    rtvDescriptorHeapDesc.NumDescriptors = 2 + mRTWritableTextures.size();
    rtvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    rtvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDescriptorHeapDesc.NodeMask = 0;
    DXCall(m_d3dDevice->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));

    // Create RTV for basic swap chain buffers
    // -----------------------------------------------
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHeapHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (int i = 0; i < 2; i++)
    {
        // Get buffer from the swap buffer
        m_SwapChain->GetBuffer(i, IID_PPV_ARGS(m_SwapChainBuffer[i].GetAddressOf()));
        // Create RTV
        m_d3dDevice->CreateRenderTargetView(m_SwapChainBuffer[i].Get(),
            nullptr,	//�ڽ������������Ѿ������˸���Դ�����ݸ�ʽ����������ָ��Ϊ��ָ��
            rtvHeapHandle);	//����������ṹ�壨�����Ǳ��壬�̳���CD3DX12_CPU_DESCRIPTOR_HANDLE��
        // Offset the rtvHeapHandle to next buffer
        rtvHeapHandle.Offset(1, m_rtvDescriptorSize);
    }

    // Create RTV for Writable Textures
    // -----------------------------------------------
    for (auto iter = mRTWritableTextures.begin(); iter != mRTWritableTextures.end(); iter++)
    {
        iter->second->CreateRtvDescriptor(rtvHeapHandle);
        rtvHeapHandle.Offset(1, m_rtvDescriptorSize);
    }

    // Set basic viewport
    // ---------------------------------
    viewPort.TopLeftX = 0;
    viewPort.TopLeftY = 0;
    viewPort.Width = width;
    viewPort.Height = height;
    viewPort.MaxDepth = 1.0f;
    viewPort.MinDepth = 0.0f;

    // Set basic scissor rectangle
    // Left-Top to Right-Bottom
    // ---------------------------------
    scissorRect.left = 0;
    scissorRect.top = 0;
    scissorRect.right = width;
    scissorRect.bottom = height;
}

void D3DModules::RTDSSubmodule::CreateDSVHeap()
{    
    //Ȼ�󴴽�DSV��
    D3D12_DESCRIPTOR_HEAP_DESC dsvDescriptorHeapDesc;
    dsvDescriptorHeapDesc.NumDescriptors = 1 + mDSWritableTextures.size();
    dsvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    dsvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvDescriptorHeapDesc.NodeMask = 0;
    ThrowIfFailed(m_d3dDevice->CreateDescriptorHeap(&dsvDescriptorHeapDesc, IID_PPV_ARGS(&m_dsvHeap)));

    // Create basic render target dsv
    // -------------------------------------------------------------------
    // 1. Filling out a D3D12_RESOURCE_DESC structure 
    //	��CPU�д��������ģ��������Դ
    D3D12_RESOURCE_DESC dsvResourceDesc;
    dsvResourceDesc.Alignment = 0;	//ָ������
    dsvResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;	//ָ����Դά�ȣ����ͣ�ΪTEXTURE2D
    dsvResourceDesc.DepthOrArraySize = 1;	//�������Ϊ1
    dsvResourceDesc.Width = width;	//��Դ��
    dsvResourceDesc.Height = height;	//��Դ��
    dsvResourceDesc.MipLevels = 1;	//MIPMAP�㼶����
    dsvResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;	//ָ�������֣����ﲻָ����
    dsvResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;	//���ģ����Դ��Flag
    dsvResourceDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;	//24λ��ȣ�8λģ��,���и������͵ĸ�ʽDXGI_FORMAT_R24G8_TYPELESSҲ����ʹ��
    dsvResourceDesc.SampleDesc.Count = 1;	//���ز�������
    dsvResourceDesc.SampleDesc.Quality = 0;	//���ز�������

    CD3DX12_CLEAR_VALUE optClear;	//�����Դ���Ż�ֵ��������������ִ���ٶȣ�CreateCommittedResource�����д��룩
    optClear.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;//24λ��ȣ�8λģ��,���и������͵ĸ�ʽDXGI_FORMAT_R24G8_TYPELESSҲ����ʹ��
    optClear.DepthStencil.Depth = 1;	//��ʼ���ֵΪ1
    optClear.DepthStencil.Stencil = 0;	//��ʼģ��ֵΪ0

    // 2. calling the ID3D12Device::CreateCommittedResource method
    //	����һ����Դ��һ���ѣ�������Դ�ύ�����У������ģ�������ύ��GPU�Դ��У�
    ThrowIfFailed(m_d3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),	//������ΪĬ�϶ѣ�����д�룩
        D3D12_HEAP_FLAG_NONE,	                //Flag
        &dsvResourceDesc,	                    //���涨���DSV��Դָ��
        D3D12_RESOURCE_STATE_COMMON,	        //��Դ��״̬Ϊ��ʼ״̬
        &optClear,	                            //���涨����Ż�ֵָ��
        IID_PPV_ARGS(&m_DepthStencilBuffer)));	//�������ģ����Դ

    m_d3dDevice->CreateDepthStencilView(m_DepthStencilBuffer.Get(),
        nullptr,	//D3D12_DEPTH_STENCIL_VIEW_DESC����ָ�룬����&dsvDesc
                    //�����ڴ������ģ����Դʱ�Ѿ��������ģ���������ԣ������������ָ��Ϊ��ָ��
        m_dsvHeap->GetCPUDescriptorHandleForHeapStart());	//DSV���

    // Transition the resource from its initial state to be used as a depth buffer.
    ptr_CommandList->ResourceBarrier(1,	//Barrier���ϸ���
        &CD3DX12_RESOURCE_BARRIER::Transition(m_DepthStencilBuffer.Get(),
            D3D12_RESOURCE_STATE_COMMON,	//ת��ǰ״̬������ʱ��״̬����CreateCommittedResource�����ж����״̬��
            D3D12_RESOURCE_STATE_DEPTH_WRITE));
    // -------------------------------------------------------------------


    // Create Writable Texture dsv
    // -------------------------------------------------------------------
    int dsvIndex = 1;
    auto dsvCpuStart = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    for (auto iter = mDSWritableTextures.begin(); iter != mDSWritableTextures.end(); iter++)
    {
        iter->second->CreateDsvDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE(dsvCpuStart, dsvIndex++, m_dsvDescriptorSize));
        // Transition the resource from its initial state to be used as a depth buffer.
        ptr_CommandList->ResourceBarrier(1,	//Barrier���ϸ���
            &CD3DX12_RESOURCE_BARRIER::Transition(iter->second->Resource(),
                D3D12_RESOURCE_STATE_COMMON,	//ת��ǰ״̬������ʱ��״̬����CreateCommittedResource�����ж����״̬��
                D3D12_RESOURCE_STATE_DEPTH_WRITE));
    }
    // -------------------------------------------------------------------
}

//======================================================================================
//======================================================================================
// ShaderResourceSubmodule
//======================================================================================
//======================================================================================

ID3D12DescriptorHeap* D3DModules::ShaderResourceSubmodule::CreateSRVHeap(std::string name, UINT num)
{
    if (num == 0)
        throw XGException("Build SRV Heap failed! Request Number 0, please check.");

    if (m_SrvHeaps.find(name) != m_SrvHeaps.end())
        throw XGException("Build SRV Heap failed! Duplicated name.");

    m_SrvHeaps[name] = nullptr;

    D3D12_DESCRIPTOR_HEAP_DESC srvDescriptorHeapDesc;
    srvDescriptorHeapDesc.NumDescriptors = num +
        MMModule->RTDSSub.mRTWritableTextures.size() +
        MMModule->RTDSSub.mDSWritableTextures.size() +
        (2 * MMModule->SRVSub.mStructuredBuffers.size()) +
        (2 * MMModule->RTDSSub.mUAWritableTextures.size());
    srvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    srvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvDescriptorHeapDesc.NodeMask = 0;
    ThrowIfFailed(m_d3dDevice->CreateDescriptorHeap(&srvDescriptorHeapDesc, IID_PPV_ARGS(&(m_SrvHeaps[name]))));

    return m_SrvHeaps[name].Get();
}

void D3DModules::ShaderResourceSubmodule::InitSRVHeap(RenderItemManager* manager)
{
    UINT offset = 0;
    CD3DX12_GPU_DESCRIPTOR_HANDLE gpuHandle(m_SrvHeaps["main"]->GetGPUDescriptorHandleForHeapStart());

    // Create SRV Descriptor for all textures
    // ------------------------------------------------
    for (auto iter = manager->mTextures.begin(); iter != manager->mTextures.end(); iter++)
    {
        std::string name = iter->first;
        // Get pointer to the start of the heap.
        CD3DX12_CPU_DESCRIPTOR_HANDLE hDescriptor(m_SrvHeaps["main"]->GetCPUDescriptorHandleForHeapStart());
        hDescriptor.Offset(manager->mTextures[name]->SrvIndex, m_cbv_srv_uavDescriptorSize);
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        D3D12_RESOURCE_DESC desc = manager->mTextures[name]->Resource->GetDesc();
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = manager->mTextures[name]->Resource->GetDesc().Format;
        switch (iter->second->type)
        {
        case Texture::Type::Texture2D:
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srvDesc.Texture2D.MostDetailedMip = 0;
            srvDesc.Texture2D.MipLevels = manager->mTextures[name]->Resource->GetDesc().MipLevels;
            srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
            break;
        case Texture::Type::Cubemap:
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            srvDesc.TextureCube.MostDetailedMip = 0;
            srvDesc.TextureCube.MipLevels = manager->mTextures[name]->Resource->GetDesc().MipLevels;
            srvDesc.TextureCube.ResourceMinLODClamp = 0.0f;
            break;
        case Texture::Type::CubemapArray:
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
            srvDesc.TextureCubeArray.MostDetailedMip = 0;
            srvDesc.TextureCubeArray.MipLevels = desc.MipLevels;
            srvDesc.TextureCubeArray.ResourceMinLODClamp = 0.0f;
            srvDesc.TextureCubeArray.First2DArrayFace = 0;
            srvDesc.TextureCubeArray.NumCubes = desc.DepthOrArraySize / 6;
            break;
        default:
            break;
        }
        m_d3dDevice->CreateShaderResourceView(manager->mTextures[name]->Resource.Get(), &srvDesc, hDescriptor);
        offset++;
        gpuHandle.Offset(1, m_cbv_srv_uavDescriptorSize);
    }

    // Create SRV Descriptor for all writable textures
    // ------------------------------------------------
    for (auto iter = MMModule->RTDSSub.mRTWritableTextures.begin(); iter != MMModule->RTDSSub.mRTWritableTextures.end(); iter++)
    {
        iter->second->SrvIndex = offset;
        iter->second->CreateSrvDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE(
            m_SrvHeaps["main"]->GetCPUDescriptorHandleForHeapStart()
            , offset++, m_cbv_srv_uavDescriptorSize), gpuHandle);
        gpuHandle.Offset(1, m_cbv_srv_uavDescriptorSize);
    }

    for (auto iter = MMModule->RTDSSub.mDSWritableTextures.begin(); iter != MMModule->RTDSSub.mDSWritableTextures.end(); iter++)
    {
        iter->second->SrvIndex = offset;
        iter->second->CreateSrvDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE(
            m_SrvHeaps["main"]->GetCPUDescriptorHandleForHeapStart()
            , offset++, m_cbv_srv_uavDescriptorSize), gpuHandle);
        gpuHandle.Offset(1, m_cbv_srv_uavDescriptorSize);
    }

    for (auto iter = MMModule->RTDSSub.mUAWritableTextures.begin(); iter != MMModule->RTDSSub.mUAWritableTextures.end(); iter++)
    {
        iter->second->SrvIndex = offset;
        iter->second->CreateSrvDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE(
            m_SrvHeaps["main"]->GetCPUDescriptorHandleForHeapStart()
            , offset++, m_cbv_srv_uavDescriptorSize), gpuHandle);
        gpuHandle.Offset(1, m_cbv_srv_uavDescriptorSize);
    }

    //for (auto iter = MMModule->SRVSub.mStructuredBuffers.begin(); iter != MMModule->SRVSub.mStructuredBuffers.end(); iter++)
    //{
    //    iter->second->SrvIndex = offset;
    //    iter->second->CreateSrvDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE(
    //        m_SrvHeaps["main"]->GetCPUDescriptorHandleForHeapStart()
    //        , offset++, m_cbv_srv_uavDescriptorSize), gpuHandle);
    //    gpuHandle.Offset(1, m_cbv_srv_uavDescriptorSize);
    //}

    // Create UAV Descriptor for all unord textures
    // ------------------------------------------------
    for (auto iter = MMModule->RTDSSub.mUAWritableTextures.begin(); iter != MMModule->RTDSSub.mUAWritableTextures.end(); iter++)
    {
        iter->second->UavIndex = offset;
        iter->second->CreateUavDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE(
            m_SrvHeaps["main"]->GetCPUDescriptorHandleForHeapStart()
            , offset++, m_cbv_srv_uavDescriptorSize), gpuHandle);
        gpuHandle.Offset(1, m_cbv_srv_uavDescriptorSize);
    }

    for (auto iter = MMModule->SRVSub.mStructuredBuffers.begin(); iter != MMModule->SRVSub.mStructuredBuffers.end(); iter++)
    {
        if (iter->second->Writable == false) continue;
        iter->second->UavIndex = offset;
        iter->second->CreateUavDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE(
            m_SrvHeaps["main"]->GetCPUDescriptorHandleForHeapStart()
            , offset++, m_cbv_srv_uavDescriptorSize), gpuHandle);
        gpuHandle.Offset(1, m_cbv_srv_uavDescriptorSize);
    }
}