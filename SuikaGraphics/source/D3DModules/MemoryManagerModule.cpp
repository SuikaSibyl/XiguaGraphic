#include <MemoryManagerModule.h>

//======================================================================================
//======================================================================================
// RenderTarget
//======================================================================================
//======================================================================================
D3DModules::RenderTarget::RenderTarget(UINT width, UINT height, DXGI_FORMAT format, UINT index, MemoryManagerModule* MMModule)
{
    // Init target features
    mTargetWidth = width;
    mTargetHeight = height;
    mTargetFormat = format;
    // Init devices objects
    device = MMModule->m_d3dDevice;
    m_rtvHeap = MMModule->RTVSub.m_rtvHeap.Get();
    m_srvHeap = MMModule->RTVSub.m_RtvSrvMainHeap;
    m_rtvDescriptorSize = MMModule->RTVSub.m_rtvDescriptorSize;

    RTVHeapIdx = index + 2;
    SRVHeapIdx = index;

    D3D12_RESOURCE_DESC hdrDesc = {};
    hdrDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    hdrDesc.Alignment = 0;
    hdrDesc.Width = mTargetWidth;
    hdrDesc.Height = mTargetHeight;
    hdrDesc.DepthOrArraySize = 1;
    hdrDesc.MipLevels = 1;
    hdrDesc.Format = mTargetFormat;
    hdrDesc.SampleDesc.Count = 1;
    hdrDesc.SampleDesc.Quality = 0;
    hdrDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    hdrDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;    // * IMPORTANT * //

    /*clear颜色与Render函数的Clear必须一致，这样一来我们即得到了驱动层的一个优化处理，也避免了在调试时，
        因为渲染循环反复执行而不断输出的一个因为两个颜色不一致，而产生的未优化警告信息。*/
    D3D12_CLEAR_VALUE optClear = {};
    optClear.Format = mTargetFormat;
    memcpy(optClear.Color, &ClearColor, 4 * sizeof(float));

    // Create Reasource
    ThrowIfFailed(device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),	//堆类型为默认堆（不能写入）
        D3D12_HEAP_FLAG_NONE,	                            //Flag
        &hdrDesc,	                                        //上面定义的DSV资源指针
        D3D12_RESOURCE_STATE_COMMON,	                    //资源的状态为初始状态
        &optClear,	                                        //上面定义的优化值指针
        IID_PPV_ARGS(&mResource)));	                    //返回深度模板资源

    //md3dDevice->CreateRenderTargetView(mHDRRendertarget->GetResource(), nullptr, rtvHeapHandle);

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHeapHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart());
    rtvHeapHandle.Offset(index + 2, m_rtvDescriptorSize);
    device->CreateRenderTargetView(mResource.Get(), nullptr, rtvHeapHandle);

    //// Get pointer to the start of the heap.
    //CD3DX12_CPU_DESCRIPTOR_HANDLE hDescriptor(m_srvHeap->GetCPUDescriptorHandleForHeapStart());
    ////hDescriptor.Offset(mTextures[name]->Index, ptr_d3dWidget->m_cbv_srv_uavDescriptorSize);
    //D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    //srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    //srvDesc.Format = mResource->GetDesc().Format;
    //srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    //srvDesc.Texture2D.MostDetailedMip = 0;
    //srvDesc.Texture2D.MipLevels = mResource->GetDesc().MipLevels;
    //srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
    //// Create Resource View
    //device->CreateShaderResourceView(mResource.Get(), &srvDesc, hDescriptor);
}

//======================================================================================
//======================================================================================
// RenderTargetSubmodule
//======================================================================================
//======================================================================================

void D3DModules::RenderTargetSubmodule::CreateRTVHeap(UINT num)
{
    D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc;
    rtvDescriptorHeapDesc.NumDescriptors = num;
    rtvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    rtvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDescriptorHeapDesc.NodeMask = 0;
    DXCall(m_d3dDevice->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHeapHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (int i = 0; i < 2; i++)
    {
        //获得存于交换链中的后台缓冲区资源
        m_SwapChain->GetBuffer(i, IID_PPV_ARGS(m_SwapChainBuffer[i].GetAddressOf()));
        //创建RTV
        m_d3dDevice->CreateRenderTargetView(m_SwapChainBuffer[i].Get(),
            nullptr,	//在交换链创建中已经定义了该资源的数据格式，所以这里指定为空指针
            rtvHeapHandle);	//描述符句柄结构体（这里是变体，继承自CD3DX12_CPU_DESCRIPTOR_HANDLE）
        //偏移到描述符堆中的下一个缓冲区
        rtvHeapHandle.Offset(1, m_rtvDescriptorSize);
    }

    if (num > 2)
    {
        for (int i = 0; i < num - 2; i++)
        {
            mRenderTarget["Assist" + std::to_string(i)] = std::make_unique<RenderTarget>
                (width, height, DXGI_FORMAT_R8G8B8A8_UNORM, i, MMModule);
        }
    }

    //然后创建DSV堆
    D3D12_DESCRIPTOR_HEAP_DESC dsvDescriptorHeapDesc;
    dsvDescriptorHeapDesc.NumDescriptors = 1;
    dsvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    dsvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvDescriptorHeapDesc.NodeMask = 0;
    ThrowIfFailed(m_d3dDevice->CreateDescriptorHeap(&dsvDescriptorHeapDesc, IID_PPV_ARGS(&m_dsvHeap)));
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

    // Set viewport
    // 视口设置
    viewPort.TopLeftX = 0;
    viewPort.TopLeftY = 0;
    viewPort.Width = width;
    viewPort.Height = height;
    viewPort.MaxDepth = 1.0f;
    viewPort.MinDepth = 0.0f;
    // Set scissor rectangle
    // 裁剪矩形设置（矩形外的像素都将被剔除）
    // 前两个为左上点坐标，后两个为右下点坐标
    scissorRect.left = 0;
    scissorRect.top = 0;
    scissorRect.right = width;
    scissorRect.bottom = height;
}