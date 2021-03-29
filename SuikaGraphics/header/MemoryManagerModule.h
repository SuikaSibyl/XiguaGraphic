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
            //接下来设置视口和裁剪矩形。
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
                // 等到渲染完成，我们要将后台缓冲区的状态改成呈现状态，使其之后推到前台缓冲区显示。完了，关闭命令列表，等待传入命令队列。
                ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(resource_addr,
                    D3D12_RESOURCE_STATE_RENDER_TARGET, 
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));//从渲染目标到呈现
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
            //接着我们将后台缓冲资源从呈现状态转换到渲染目标状态（即准备接收图像渲染）。
            ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(resource_addr,//转换资源为后台缓冲区资源
                prev_state, D3D12_RESOURCE_STATE_RENDER_TARGET));//从呈现到渲染目标转换
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

            /*clear颜色与Render函数的Clear必须一致，这样一来我们即得到了驱动层的一个优化处理，也避免了在调试时，
                因为渲染循环反复执行而不断输出的一个因为两个颜色不一致，而产生的未优化警告信息。*/
            D3D12_CLEAR_VALUE optClear = {};
            float dark[4] = { 0.117,0.117,0.117,1 };
            optClear.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            memcpy(optClear.Color, &dark, 4 * sizeof(float));

            DXCall(m_d3dDevice->CreateCommittedResource(
                &readbackHeapProperties,
                D3D12_HEAP_FLAG_NONE,
                &bufferDesc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,	                                        //上面定义的优化值指针
                IID_PPV_ARGS(&readbackBuffer)));	                    //返回深度模板资源

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
            // 等到渲染完成，我们要将后台缓冲区的状态改成呈现状态，使其之后推到前台缓冲区显示。完了，关闭命令列表，等待传入命令队列。
            ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_SwapChainBuffer[m_CurrentBackBuffer].Get(),
                D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));//从渲染目标到呈现
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
            //    &readbackHeapProperties, //读回堆类型
            //    D3D12_HEAP_FLAG_NONE,
            //    &readbackBufferDesc, //默认堆为最终存储数据的地方，所以暂时初始化为普通状态
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