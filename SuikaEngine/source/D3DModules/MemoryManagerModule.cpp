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
    //接着我们将后台缓冲资源从呈现状态转换到渲染目标状态（即准备接收图像渲染）。
    ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(resource_addr,//转换资源为后台缓冲区资源
        screenRTState, D3D12_RESOURCE_STATE_RENDER_TARGET));//从呈现到渲染目标转换
    screenRTState = D3D12_RESOURCE_STATE_RENDER_TARGET;
    // Clear the back buffer and depth buffer.
    //然后清除后台缓冲区和深度缓冲区，并赋值。步骤是先获得堆中描述符句柄（即地址），再通过ClearRenderTargetView函数和ClearDepthStencilView函数做清除和赋值。这里我们将RT资源背景色赋值为DarkRed（暗红）。
    float dark[4] = { 0.117,0.117,0.117,1 };
    ptr_CommandList->ClearRenderTargetView(rtvHandle, dark, 0, nullptr);//清除RT背景色为暗红，并且不设置裁剪矩形

    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    ptr_CommandList->ClearDepthStencilView(dsvHandle,	//DSV描述符句柄
        D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
        1.0f,	//默认深度值
        0,	//默认模板值
        0,	//裁剪矩形数量
        nullptr);	//裁剪矩形指针

    // Specify the buffers we are going to render to. 
    //然后我们指定将要渲染的缓冲区，即指定RTV和DSV。
    ptr_CommandList->OMSetRenderTargets(1,//待绑定的RTV数量
        &rtvHandle,	//指向RTV数组的指针
        true,	//RTV对象在堆内存中是连续存放的
        &dsvHandle);	//指向DSV的指针
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
    //设置Viewports与ScissorRects
    ptr_CommandList->RSSetViewports(1, &mRTWritableTextures[name]->Viewport());
    ptr_CommandList->RSSetScissorRects(1, &mRTWritableTextures[name]->ScissorRect());
    // Indicate a state transition on the resource usage.
    //接着我们将后台缓冲资源从呈现状态转换到渲染目标状态（即准备接收图像渲染）。
    mRTWritableTextures[name]->ChangeResourceState(ptr_CommandList, D3D12_RESOURCE_STATE_RENDER_TARGET);
    //ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(resource_addr,//转换资源为后台缓冲区资源
    //    prev_state, D3D12_RESOURCE_STATE_RENDER_TARGET));//从呈现到渲染目标转换
    // Clear the back buffer and depth buffer.
    //然后清除后台缓冲区和深度缓冲区，并赋值。步骤是先获得堆中描述符句柄（即地址），再通过ClearRenderTargetView函数和ClearDepthStencilView函数做清除和赋值。这里我们将RT资源背景色赋值为DarkRed（暗红）。
    float dark[4] = { 0.117,0.117,0.117,1 };
    ptr_CommandList->ClearRenderTargetView(rtvHandle, dark, 0, nullptr);//清除RT背景色为暗红，并且不设置裁剪矩形

    // Set DSV
    // ------------------------------------------
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    if (dsname != "DEFAULT")
    {
        dsvHandle = mDSWritableTextures[dsname]->Dsv();
    }
    ptr_CommandList->ClearDepthStencilView(dsvHandle,	//DSV描述符句柄
        D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
        1.0f,	//默认深度值
        0,	//默认模板值
        0,	//裁剪矩形数量
        nullptr);	//裁剪矩形指针

    // Specify the buffers we are going to render to. 
    //然后我们指定将要渲染的缓冲区，即指定RTV和DSV。
    ptr_CommandList->OMSetRenderTargets(1,//待绑定的RTV数量
        &rtvHandle,	//指向RTV数组的指针
        true,	//RTV对象在堆内存中是连续存放的
        &dsvHandle);	//指向DSV的指针
}

void D3DModules::RTDSSubmodule::RenderToTexture(std::string name, std::string name2, std::string dsname)
{
    //// Set RTV
    //// ------------------------------------------
    //resource_addr = mRTWritableTextures[name]->Resource();
    ////设置Viewports与ScissorRects
    //ptr_CommandList->RSSetViewports(1, &mRTWritableTextures[name]->Viewport());
    //ptr_CommandList->RSSetScissorRects(1, &mRTWritableTextures[name]->ScissorRect());
    //// Indicate a state transition on the resource usage.
    ////接着我们将后台缓冲资源从呈现状态转换到渲染目标状态（即准备接收图像渲染）。
    //mRTWritableTextures[name]->ChangeResourceState(ptr_CommandList, D3D12_RESOURCE_STATE_RENDER_TARGET);
    //mRTWritableTextures[name2]->ChangeResourceState(ptr_CommandList, D3D12_RESOURCE_STATE_RENDER_TARGET);
    ////ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(resource_addr,//转换资源为后台缓冲区资源
    ////    prev_state, D3D12_RESOURCE_STATE_RENDER_TARGET));//从呈现到渲染目标转换
    //// Clear the back buffer and depth buffer.
    ////然后清除后台缓冲区和深度缓冲区，并赋值。步骤是先获得堆中描述符句柄（即地址），再通过ClearRenderTargetView函数和ClearDepthStencilView函数做清除和赋值。这里我们将RT资源背景色赋值为DarkRed（暗红）。
    //float dark[4] = { 0.117,0.117,0.117,1 };
    //ptr_CommandList->ClearRenderTargetView(rtvHandle, dark, 0, nullptr);//清除RT背景色为暗红，并且不设置裁剪矩形
    ////ptr_CommandList->ClearRenderTargetView(rtv2Handle, dark, 0, nullptr);//清除RT背景色为暗红，并且不设置裁剪矩形

    //// Set DSV
    //// ------------------------------------------
    //D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    //if (dsname != "DEFAULT")
    //{
    //    dsvHandle = mDSWritableTextures[dsname]->Dsv();
    //}
    //ptr_CommandList->ClearDepthStencilView(dsvHandle,	//DSV描述符句柄
    //    D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
    //    1.0f,	//默认深度值
    //    0,	//默认模板值
    //    0,	//裁剪矩形数量
    //    nullptr);	//裁剪矩形指针

    //D3D12_CPU_DESCRIPTOR_HANDLE* handle_array = new D3D12_CPU_DESCRIPTOR_HANDLE();
    //handle_array[0] = rtvHandle;
    //handle_array[1] = rtv2Handle;
    //// Specify the buffers we are going to render to. 
    ////然后我们指定将要渲染的缓冲区，即指定RTV和DSV。
    //ptr_CommandList->OMSetRenderTargets(1,//待绑定的RTV数量
    //    &rtvHandle,	//指向RTV数组的指针
    //    true,	//RTV对象在堆内存中是连续存放的
    //    &dsvHandle);	//指向DSV的指针


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
    //设置Viewports与ScissorRects
    ptr_CommandList->RSSetViewports(1, &mRTWritableTextures[name]->Viewport());
    ptr_CommandList->RSSetScissorRects(1, &mRTWritableTextures[name]->ScissorRect());
    // Indicate a state transition on the resource usage.
    //接着我们将后台缓冲资源从呈现状态转换到渲染目标状态（即准备接收图像渲染）。
    mRTWritableTextures[name]->ChangeResourceState(ptr_CommandList, D3D12_RESOURCE_STATE_RENDER_TARGET);
    mRTWritableTextures[name2]->ChangeResourceState(ptr_CommandList, D3D12_RESOURCE_STATE_RENDER_TARGET);
    //ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(resource_addr,//转换资源为后台缓冲区资源
    //    prev_state, D3D12_RESOURCE_STATE_RENDER_TARGET));//从呈现到渲染目标转换
    // Clear the back buffer and depth buffer.
    //然后清除后台缓冲区和深度缓冲区，并赋值。步骤是先获得堆中描述符句柄（即地址），再通过ClearRenderTargetView函数和ClearDepthStencilView函数做清除和赋值。这里我们将RT资源背景色赋值为DarkRed（暗红）。
    float dark[4] = { 0.117,0.117,0.117,1 };
    ptr_CommandList->ClearRenderTargetView(rtvHandle, dark, 0, nullptr);//清除RT背景色为暗红，并且不设置裁剪矩形
    ptr_CommandList->ClearRenderTargetView(rtv2Handle, dark, 0, nullptr);//清除RT背景色为暗红，并且不设置裁剪矩形

    // Set DSV
    // ------------------------------------------
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    if (dsname != "DEFAULT")
    {
        dsvHandle = mDSWritableTextures[dsname]->Dsv();
    }
    ptr_CommandList->ClearDepthStencilView(dsvHandle,	//DSV描述符句柄
        D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
        1.0f,	//默认深度值
        0,	//默认模板值
        0,	//裁剪矩形数量
        nullptr);	//裁剪矩形指针

    D3D12_CPU_DESCRIPTOR_HANDLE* handle_array = new D3D12_CPU_DESCRIPTOR_HANDLE[2];
    handle_array[0] = rtvHandle;
    handle_array[1] = rtv2Handle;

    // Specify the buffers we are going to render to. 
    //然后我们指定将要渲染的缓冲区，即指定RTV和DSV。
    ptr_CommandList->OMSetRenderTargets(2,//待绑定的RTV数量
        handle_array,	//指向RTV数组的指针
        false,	//RTV对象在堆内存中是连续存放的
        &dsvHandle);	//指向DSV的指针
}
void D3DModules::RTDSSubmodule::CleanPreviousRenderTarget()
{
    //// 等到渲染完成，我们要将后台缓冲区的状态改成呈现状态，使其之后推到前台缓冲区显示。完了，关闭命令列表，等待传入命令队列。
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

    //将后台缓冲资源转成“复制目标”
    ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        screen_addr,
        screenRTState,
        D3D12_RESOURCE_STATE_COPY_DEST));
    screenRTState = D3D12_RESOURCE_STATE_COPY_DEST;

    //将模糊处理后的离屏纹理拷贝给后台缓冲区
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

    //将后台缓冲资源转成“复制目标”
    ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        screen_addr,
        screenRTState,
        D3D12_RESOURCE_STATE_COPY_DEST));
    screenRTState = D3D12_RESOURCE_STATE_COPY_DEST;

    //将模糊处理后的离屏纹理拷贝给后台缓冲区
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
            nullptr,	//在交换链创建中已经定义了该资源的数据格式，所以这里指定为空指针
            rtvHeapHandle);	//描述符句柄结构体（这里是变体，继承自CD3DX12_CPU_DESCRIPTOR_HANDLE）
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
    //然后创建DSV堆
    D3D12_DESCRIPTOR_HEAP_DESC dsvDescriptorHeapDesc;
    dsvDescriptorHeapDesc.NumDescriptors = 1 + mDSWritableTextures.size();
    dsvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    dsvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvDescriptorHeapDesc.NodeMask = 0;
    ThrowIfFailed(m_d3dDevice->CreateDescriptorHeap(&dsvDescriptorHeapDesc, IID_PPV_ARGS(&m_dsvHeap)));

    // Create basic render target dsv
    // -------------------------------------------------------------------
    // 1. Filling out a D3D12_RESOURCE_DESC structure 
    //	在CPU中创建好深度模板数据资源
    D3D12_RESOURCE_DESC dsvResourceDesc;
    dsvResourceDesc.Alignment = 0;	//指定对齐
    dsvResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;	//指定资源维度（类型）为TEXTURE2D
    dsvResourceDesc.DepthOrArraySize = 1;	//纹理深度为1
    dsvResourceDesc.Width = width;	//资源宽
    dsvResourceDesc.Height = height;	//资源高
    dsvResourceDesc.MipLevels = 1;	//MIPMAP层级数量
    dsvResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;	//指定纹理布局（这里不指定）
    dsvResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;	//深度模板资源的Flag
    dsvResourceDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;	//24位深度，8位模板,还有个无类型的格式DXGI_FORMAT_R24G8_TYPELESS也可以使用
    dsvResourceDesc.SampleDesc.Count = 1;	//多重采样数量
    dsvResourceDesc.SampleDesc.Quality = 0;	//多重采样质量

    CD3DX12_CLEAR_VALUE optClear;	//清除资源的优化值，提高清除操作的执行速度（CreateCommittedResource函数中传入）
    optClear.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;//24位深度，8位模板,还有个无类型的格式DXGI_FORMAT_R24G8_TYPELESS也可以使用
    optClear.DepthStencil.Depth = 1;	//初始深度值为1
    optClear.DepthStencil.Stencil = 0;	//初始模板值为0

    // 2. calling the ID3D12Device::CreateCommittedResource method
    //	创建一个资源和一个堆，并将资源提交至堆中（将深度模板数据提交至GPU显存中）
    ThrowIfFailed(m_d3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),	//堆类型为默认堆（不能写入）
        D3D12_HEAP_FLAG_NONE,	                //Flag
        &dsvResourceDesc,	                    //上面定义的DSV资源指针
        D3D12_RESOURCE_STATE_COMMON,	        //资源的状态为初始状态
        &optClear,	                            //上面定义的优化值指针
        IID_PPV_ARGS(&m_DepthStencilBuffer)));	//返回深度模板资源

    m_d3dDevice->CreateDepthStencilView(m_DepthStencilBuffer.Get(),
        nullptr,	//D3D12_DEPTH_STENCIL_VIEW_DESC类型指针，可填&dsvDesc
                    //由于在创建深度模板资源时已经定义深度模板数据属性，所以这里可以指定为空指针
        m_dsvHeap->GetCPUDescriptorHandleForHeapStart());	//DSV句柄

    // Transition the resource from its initial state to be used as a depth buffer.
    ptr_CommandList->ResourceBarrier(1,	//Barrier屏障个数
        &CD3DX12_RESOURCE_BARRIER::Transition(m_DepthStencilBuffer.Get(),
            D3D12_RESOURCE_STATE_COMMON,	//转换前状态（创建时的状态，即CreateCommittedResource函数中定义的状态）
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
        ptr_CommandList->ResourceBarrier(1,	//Barrier屏障个数
            &CD3DX12_RESOURCE_BARRIER::Transition(iter->second->Resource(),
                D3D12_RESOURCE_STATE_COMMON,	//转换前状态（创建时的状态，即CreateCommittedResource函数中定义的状态）
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