#pragma once
#include <string>
#include <Utility.h>
#include <RenderItemManagment.h>

using Microsoft::WRL::ComPtr;
namespace D3DModules
{
    class MemoryManagerModule;
    class ShaderResourceSubmodule;

    class RenderTarget
    {
    public:
        friend class ShaderResourceSubmodule;

        RenderTarget(UINT width, UINT height, DXGI_FORMAT format, UINT index, MemoryManagerModule* MMModule);
        ID3D12Resource* Resource() { return mResource.Get(); }

    private:
        UINT mTargetWidth;
        UINT mTargetHeight;
        DXGI_FORMAT mTargetFormat;
        ComPtr<ID3D12Resource> mResource = nullptr;

        ID3D12Device* device = nullptr;
        ID3D12DescriptorHeap* m_rtvHeap = nullptr;
        ID3D12DescriptorHeap* m_srvHeap = nullptr;

        const float ClearColor[4] = { 0.117,0.117,0.117,1 };
        UINT  m_rtvDescriptorSize;

        UINT RTVHeapIdx = 0;
        UINT SRVHeapIdx = 0;
    };

    //===================================================================================
    //===================================================================================
    // RENDER TARGET SUBMODULE
    //===================================================================================
    //===================================================================================
    class RenderTargetSubmodule
    {
    public:
        // Frineds
        // -------------------------------
        friend class RenderTarget;
        friend class MemoryManagerModule;
        friend class ShaderResourceSubmodule;

        RenderTargetSubmodule(ID3D12Device* device, MemoryManagerModule* mmmodule) :m_d3dDevice(device)
        {
            MMModule = mmmodule;
            m_rtvDescriptorSize = m_d3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        }

        void CreateRTVHeap(UINT num);

        void SetSwapChain(IDXGISwapChain* swapChain)
        {
            m_SwapChain = swapChain;
        }

        void SetWidthHeight(float w, float h)
        {
            width = w;
            height = h;
        }

        void StartNewFrame()
        {
            //�����������ӿںͲü����Ρ�
            ptr_CommandList->RSSetViewports(1, &viewPort);
            ptr_CommandList->RSSetScissorRects(1, &scissorRect);
        }

        void ResetRenderTarget(bool RenderToScreen = true, UINT index = 0)
        {
            static UINT previous = -1;

            // Reset previous RenderTarget Mode to COMMON
            if (previous != -1)
            {
                ID3D12Resource* resource_addr = mRenderTarget["Assist" + std::to_string(previous)]->Resource();
                // �ȵ���Ⱦ��ɣ�����Ҫ����̨��������״̬�ĳɳ���״̬��ʹ��֮���Ƶ�ǰ̨��������ʾ�����ˣ��ر������б��ȴ�����������С�
                ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(resource_addr,
                    D3D12_RESOURCE_STATE_RENDER_TARGET, 
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));//����ȾĿ�굽����
            }

            // Set the state of new Render Target
            D3D12_RESOURCE_STATES prev_state;
            ID3D12Resource* resource_addr;
            D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
            if (RenderToScreen)
            {
                previous = -1;
                prev_state = D3D12_RESOURCE_STATE_PRESENT;
                resource_addr = m_SwapChainBuffer[m_CurrentBackBuffer].Get();
                rtvHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), m_CurrentBackBuffer, m_rtvDescriptorSize);
            }
            else
            {
                previous = index;
                prev_state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                resource_addr = mRenderTarget["Assist" + std::to_string(index)]->Resource();
                rtvHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), 2, m_rtvDescriptorSize);
            }

            // Indicate a state transition on the resource usage.
            //�������ǽ���̨������Դ�ӳ���״̬ת������ȾĿ��״̬����׼������ͼ����Ⱦ����
            ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(resource_addr,//ת����ԴΪ��̨��������Դ
                prev_state, D3D12_RESOURCE_STATE_RENDER_TARGET));//�ӳ��ֵ���ȾĿ��ת��
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

        void ReadBackToBuffer()
        {
            ID3D12Resource* resource = mRenderTarget["Assist0"]->Resource();

            D3D12_HEAP_PROPERTIES readbackHeapProperties{ CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK) };
            //D3D12_RESOURCE_DESC readbackBufferDesc{ resource->GetDesc() };
            ComPtr<ID3D12Resource> readbackBuffer;

            D3D12_RESOURCE_DESC bufferDesc = {};
            bufferDesc.Alignment = resource->GetDesc().Alignment;
            bufferDesc.DepthOrArraySize = 1;
            bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
            bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
            bufferDesc.Height = 1;
            bufferDesc.Width = resource->GetDesc().Width * resource->GetDesc().Height;
            bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            bufferDesc.MipLevels = 1;
            bufferDesc.SampleDesc.Count = 1;
            bufferDesc.SampleDesc.Quality = 0;

            /*clear��ɫ��Render������Clear����һ�£�����һ�����Ǽ��õ����������һ���Ż�����Ҳ�������ڵ���ʱ��
                ��Ϊ��Ⱦѭ������ִ�ж����������һ����Ϊ������ɫ��һ�£���������δ�Ż�������Ϣ��*/
            D3D12_CLEAR_VALUE optClear = {};
            float dark[4] = { 0.117,0.117,0.117,1 };
            optClear.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            memcpy(optClear.Color, &dark, 4 * sizeof(float));

            DXCall(m_d3dDevice->CreateCommittedResource(
                &readbackHeapProperties,
                D3D12_HEAP_FLAG_NONE,
                &bufferDesc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,	                                        //���涨����Ż�ֵָ��
                IID_PPV_ARGS(&readbackBuffer)));	                    //�������ģ����Դ

            {
                D3D12_RESOURCE_BARRIER outputBufferResourceBarrier
                {
                    CD3DX12_RESOURCE_BARRIER::Transition(
                        resource,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COPY_SOURCE)
                };
                ptr_CommandList->ResourceBarrier(1, &outputBufferResourceBarrier);
            }
            ptr_CommandList->CopyResource(readbackBuffer.Get(), resource);

            {
                D3D12_RESOURCE_BARRIER outputBufferResourceBarrier
                {
                    CD3DX12_RESOURCE_BARRIER::Transition(
                        resource,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
                };
                ptr_CommandList->ResourceBarrier(1, &outputBufferResourceBarrier);
            }
            // Code goes here to close, execute (and optionally reset) the command list, and also
            // to use a fence to wait for the command queue.

            // The code below assumes that the GPU wrote FLOATs to the buffer.
            D3D12_RANGE readbackBufferRange{ 0, resource->GetDesc().Width * resource->GetDesc().Height };
            FLOAT* pReadbackBufferData{};
            DXCall(
                readbackBuffer->Map
                (
                    0,
                    &readbackBufferRange,
                    reinterpret_cast<void**>(&pReadbackBufferData)
                )
            );
        }
        void EndNewFrame()
        {
            // Indicate a state transition on the resource usage.
            // �ȵ���Ⱦ��ɣ�����Ҫ����̨��������״̬�ĳɳ���״̬��ʹ��֮���Ƶ�ǰ̨��������ʾ�����ˣ��ر������б��ȴ�����������С�
            ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_SwapChainBuffer[m_CurrentBackBuffer].Get(),
                D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));//����ȾĿ�굽����
            m_CurrentBackBuffer = (m_CurrentBackBuffer + 1) % 2;
        }
    private:
        // Device Objects
        ID3D12Device* m_d3dDevice;
        // 
        IDXGISwapChain* m_SwapChain;
        ComPtr<ID3D12DescriptorHeap>    m_rtvHeap = nullptr;
        ComPtr<ID3D12DescriptorHeap>    m_dsvHeap = nullptr;
        ComPtr<ID3D12Resource>          m_SwapChainBuffer[2];
        ComPtr<ID3D12Resource>          m_DepthStencilBuffer;
        std::unordered_map<std::string, std::unique_ptr<RenderTarget>> mRenderTarget;
        UINT  m_CurrentBackBuffer = 0;
        UINT  m_rtvDescriptorSize;
        // Rect & ViewPort
        D3D12_RECT scissorRect;
        D3D12_VIEWPORT viewPort;
        // Size
        float width, height;
        ID3D12DescriptorHeap* m_RtvSrvMainHeap = nullptr;

        MemoryManagerModule* MMModule;
        ID3D12GraphicsCommandList* ptr_CommandList;
    };

    //===================================================================================
    //===================================================================================
    // SHADER RESOURCE SUBMODULE
    //===================================================================================
    //===================================================================================
    class ShaderResourceSubmodule
    {
    public:
        friend class MemoryManagerModule;

        ShaderResourceSubmodule(ID3D12Device* device, MemoryManagerModule* mmmodule) :m_d3dDevice(device)
        {
            MMModule = mmmodule;
            m_cbv_srv_uavDescriptorSize = m_d3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }

        ID3D12DescriptorHeap* CreateSRVHeap(std::string name, UINT num)
        {
            if (num == 0)
                throw XGException("Build SRV Heap failed! Request Number 0, please check.");

            if (m_SrvHeaps.find(name) != m_SrvHeaps.end())
                throw XGException("Build SRV Heap failed! Duplicated name.");

            m_SrvHeaps[name] = nullptr;

            D3D12_DESCRIPTOR_HEAP_DESC srvDescriptorHeapDesc;
            srvDescriptorHeapDesc.NumDescriptors = num;
            srvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            srvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            srvDescriptorHeapDesc.NodeMask = 0;
            ThrowIfFailed(m_d3dDevice->CreateDescriptorHeap(&srvDescriptorHeapDesc, IID_PPV_ARGS(&(m_SrvHeaps[name]))));

            return m_SrvHeaps[name].Get();
        }

        void InitSRVHeap(RenderItemManager* manager, RenderTargetSubmodule* RTSub)
        {
            UINT offset = 0;
            for (auto iter = manager->mTextures.begin(); iter != manager->mTextures.end(); iter++)
            {
                std::string name = iter->first;
                // Get pointer to the start of the heap.
                CD3DX12_CPU_DESCRIPTOR_HANDLE hDescriptor(m_SrvHeaps["main"]->GetCPUDescriptorHandleForHeapStart());
                hDescriptor.Offset(manager->mTextures[name]->Index, m_cbv_srv_uavDescriptorSize);
                D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
                srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srvDesc.Format = manager->mTextures[name]->Resource->GetDesc().Format;
                srvDesc.ViewDimension = iter->second->isCubeMap ? D3D12_SRV_DIMENSION_TEXTURECUBE : D3D12_SRV_DIMENSION_TEXTURE2D;
                srvDesc.Texture2D.MostDetailedMip = 0;
                srvDesc.Texture2D.MipLevels = manager->mTextures[name]->Resource->GetDesc().MipLevels;
                srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
                m_d3dDevice->CreateShaderResourceView(manager->mTextures[name]->Resource.Get(), &srvDesc, hDescriptor);
                offset++;
            }

            for (auto iter = RTSub->mRenderTarget.begin(); iter != RTSub->mRenderTarget.end(); iter++)
            {
                std::string name = iter->first;
                // Get pointer to the start of the heap.
                CD3DX12_CPU_DESCRIPTOR_HANDLE hDescriptor(m_SrvHeaps["main"]->GetCPUDescriptorHandleForHeapStart());
                iter->second->SRVHeapIdx += offset;
                hDescriptor.Offset(iter->second->SRVHeapIdx, m_cbv_srv_uavDescriptorSize);
                D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
                srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srvDesc.Format = iter->second->mResource->GetDesc().Format;
                srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                srvDesc.Texture2D.MostDetailedMip = 0;
                srvDesc.Texture2D.MipLevels = iter->second->mResource->GetDesc().MipLevels;
                srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
                m_d3dDevice->CreateShaderResourceView(iter->second->mResource.Get(), &srvDesc, hDescriptor);
            }
        }

        ID3D12DescriptorHeap* GetMainHeap()
        {
            return m_SrvHeaps["main"].Get();
        }

    private:
        UINT m_cbv_srv_uavDescriptorSize;
        ID3D12Device* m_d3dDevice;
        MemoryManagerModule* MMModule;
        ID3D12GraphicsCommandList* ptr_CommandList;
        std::unordered_map<std::string, ComPtr<ID3D12DescriptorHeap>> m_SrvHeaps;
    };

    //===================================================================================
    //===================================================================================
    // MEMORY MANAGER MODULE
    //===================================================================================
    //===================================================================================
    class MemoryManagerModule
    {
    public:
        friend class RenderTarget;

        MemoryManagerModule(ID3D12Device* device) :
            m_d3dDevice(device),
            RTVSub(device, this),
            SRVSub(device, this)
        {}
        void SetRenderTargetNum(UINT num = 2) { RenderTargetNum = num; }

        void CreateReadbackBuffer(UINT64 outputBufferSize)
        {
            //// The output buffer (created below) is on a default heap, so only the GPU can access it.
            //D3D12_HEAP_PROPERTIES defaultHeapProperties{ CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT) };
            //D3D12_RESOURCE_DESC outputBufferDesc{ CD3DX12_RESOURCE_DESC::Buffer(outputBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) };
            //ComPtr<ID3D12Resource> outputBuffer;
            //DXCall(m_d3dDevice->CreateCommittedResource(
            //    &defaultHeapProperties,
            //    D3D12_HEAP_FLAG_NONE,
            //    &outputBufferDesc,
            //    D3D12_RESOURCE_STATE_COPY_DEST,
            //    nullptr,
            //    IID_PPV_ARGS(outputBuffer.GetAddressOf())));

            //D3D12_HEAP_PROPERTIES readbackHeapProperties{ CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK) };
            //D3D12_RESOURCE_DESC readbackBufferDesc{ CD3DX12_RESOURCE_DESC::Buffer(outputBufferSize) };
            //ComPtr<ID3D12Resource> readbackBuffer;
            //DXCall(m_d3dDevice->CreateCommittedResource(
            //    &readbackHeapProperties, //���ض�����
            //    D3D12_HEAP_FLAG_NONE,
            //    &readbackBufferDesc, //Ĭ�϶�Ϊ���մ洢���ݵĵط���������ʱ��ʼ��Ϊ��ͨ״̬
            //    D3D12_RESOURCE_STATE_COPY_DEST,
            //    nullptr,
            //    IID_PPV_ARGS(readbackBuffer.GetAddressOf())));

            //{
            //    D3D12_RESOURCE_BARRIER outputBufferResourceBarrier
            //    {
            //        CD3DX12_RESOURCE_BARRIER::Transition(
            //            outputBuffer.get(),
            //            D3D12_RESOURCE_STATE_COPY_DEST,
            //            D3D12_RESOURCE_STATE_COPY_SOURCE)
            //    };
            //    commandList->ResourceBarrier(1, &outputBufferResourceBarrier);
            //}
        }

        void CreateRTVHeap(float width, float height, IDXGISwapChain* swapChain)
        {
            RTVSub.SetSwapChain(swapChain);
            RTVSub.SetWidthHeight(width, height);
            RTVSub.CreateRTVHeap(RenderTargetNum);
        }

        void StartNewFrame()
        {
            RTVSub.StartNewFrame();
        }

        void GrabScreen()
        {
            RTVSub.ReadBackToBuffer();
        }

        void ResetRenderTarget(bool RenderToScreen, UINT index = -1)
        {
            RTVSub.ResetRenderTarget(RenderToScreen, index);
        }

        void EndNewFrame()
        {
            RTVSub.EndNewFrame();
        }

        void SetCommandList(ID3D12GraphicsCommandList* cmdList)
        {
            ptr_CommandList = cmdList;
            RTVSub.ptr_CommandList = cmdList;
            SRVSub.ptr_CommandList = cmdList;
        }

        void InitSRVHeap(RenderItemManager* manager)
        {
            SRVSub.CreateSRVHeap("main", manager->mTextures.size() + RenderTargetNum - 2);
            SRVSub.InitSRVHeap(manager, &RTVSub);
        }

        ID3D12DescriptorHeap* GetMainHeap()
        {
            return SRVSub.GetMainHeap();
        }

        RenderTargetSubmodule RTVSub;
        ShaderResourceSubmodule SRVSub;

    private:
        ID3D12Device* m_d3dDevice;
        UINT RenderTargetNum = 2;
        ID3D12GraphicsCommandList* ptr_CommandList;
    };
}