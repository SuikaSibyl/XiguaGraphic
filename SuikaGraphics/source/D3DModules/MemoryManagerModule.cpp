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

    /*clear��ɫ��Render������Clear����һ�£�����һ�����Ǽ��õ����������һ���Ż�����Ҳ�������ڵ���ʱ��
        ��Ϊ��Ⱦѭ������ִ�ж����������һ����Ϊ������ɫ��һ�£���������δ�Ż�������Ϣ��*/
    D3D12_CLEAR_VALUE optClear = {};
    optClear.Format = mTargetFormat;
    memcpy(optClear.Color, &ClearColor, 4 * sizeof(float));

    // Create Reasource
    ThrowIfFailed(device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),	//������ΪĬ�϶ѣ�����д�룩
        D3D12_HEAP_FLAG_NONE,	                            //Flag
        &hdrDesc,	                                        //���涨���DSV��Դָ��
        D3D12_RESOURCE_STATE_COMMON,	                    //��Դ��״̬Ϊ��ʼ״̬
        &optClear,	                                        //���涨����Ż�ֵָ��
        IID_PPV_ARGS(&mResource)));	                    //�������ģ����Դ

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
        //��ô��ڽ������еĺ�̨��������Դ
        m_SwapChain->GetBuffer(i, IID_PPV_ARGS(m_SwapChainBuffer[i].GetAddressOf()));
        //����RTV
        m_d3dDevice->CreateRenderTargetView(m_SwapChainBuffer[i].Get(),
            nullptr,	//�ڽ������������Ѿ������˸���Դ�����ݸ�ʽ����������ָ��Ϊ��ָ��
            rtvHeapHandle);	//����������ṹ�壨�����Ǳ��壬�̳���CD3DX12_CPU_DESCRIPTOR_HANDLE��
        //ƫ�Ƶ����������е���һ��������
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

    //Ȼ�󴴽�DSV��
    D3D12_DESCRIPTOR_HEAP_DESC dsvDescriptorHeapDesc;
    dsvDescriptorHeapDesc.NumDescriptors = 1;
    dsvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    dsvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvDescriptorHeapDesc.NodeMask = 0;
    ThrowIfFailed(m_d3dDevice->CreateDescriptorHeap(&dsvDescriptorHeapDesc, IID_PPV_ARGS(&m_dsvHeap)));
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

    // Set viewport
    // �ӿ�����
    viewPort.TopLeftX = 0;
    viewPort.TopLeftY = 0;
    viewPort.Width = width;
    viewPort.Height = height;
    viewPort.MaxDepth = 1.0f;
    viewPort.MinDepth = 0.0f;
    // Set scissor rectangle
    // �ü��������ã�����������ض������޳���
    // ǰ����Ϊ���ϵ����꣬������Ϊ���µ�����
    scissorRect.left = 0;
    scissorRect.top = 0;
    scissorRect.right = width;
    scissorRect.bottom = height;
}