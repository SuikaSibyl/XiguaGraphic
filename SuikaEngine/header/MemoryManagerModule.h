#pragma once
#include <string>
#include <Utility.h>
#include <RenderItemManagment.h>
#include <SynchronizationModule.h>
#include <Texture.h>

using Microsoft::WRL::ComPtr;
namespace D3DModules
{
    class MemoryManagerModule;
    class ShaderResourceSubmodule;

    //===================================================================================
    //===================================================================================
    // RENDER TARGET SUBMODULE
    //===================================================================================
    //===================================================================================
    class RTDSSubmodule
    {
    public:
        // Frineds
        // -------------------------------
        friend class MemoryManagerModule;
        friend class ShaderResourceSubmodule;


    public:
        // Share Functions (Basic + Writable)
        // ---------------------------------------
        RTDSSubmodule(ID3D12Device* device, MemoryManagerModule* mmmodule);
        // Create RTV Heap & DSV Heap
        void CreateRTDSHeap(float width, float height, IDXGISwapChain* swapChain);

    private:
        void CreateRTVHeap();
        void CreateDSVHeap();

    public:
        // Writbale Addictive Objects
        // ---------------------------------------
        WritableTexture* CreateWritableTexture(std::string name, UINT width, UINT height, WritableTexture::WritableType type);
        // Print a RenderTarget Texture
        void PrintWirtableTexture(std::string name, const wchar_t* file, ID3D12CommandQueue* ptrCmdQueue);
        WritableTexture* GetWrtiableTexture(std::string name, WritableTexture::WritableType type)
        {
            switch (type)
            {
            case WritableTexture::RenderTarget:
                return mRTWritableTextures[name].get();
                break;
            case WritableTexture::DepthStencil:
                return mDSWritableTextures[name].get();
                break;
            case WritableTexture::UnorderedAccess:
                return mUAWritableTextures[name].get();
                break;
            default:
                break;
            }
        }
        void SetViewportsScissor()
        {
            //接下来设置视口和裁剪矩形。
            ptr_CommandList->RSSetViewports(1, &viewPort);
            ptr_CommandList->RSSetScissorRects(1, &scissorRect);
        }

        void RenderToScreen();
        void RenderToTexture(std::string rtname, std::string dsname = "DEFAULT");
        void RenderTextureToScreen(std::string rtname);
        void UnorderedAccessTextureToScreen(std::string rtname);

        void ReadBackToBuffer()
        {
            static bool firsttime = true;
            
            if (firsttime == true)
            {
                firsttime = false;
                //mRenderTarget["Assist0"]->CaptureTexture(ptr_CommandList);
            }
        }

        void EndNewFrame()
        {
            // Indicate a state transition on the resource usage.
            // 等到渲染完成，我们要将后台缓冲区的状态改成呈现状态，使其之后推到前台缓冲区显示。完了，关闭命令列表，等待传入命令队列。
            ptr_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_SwapChainBuffer[m_CurrentBackBuffer].Get(),
                screenRTState, D3D12_RESOURCE_STATE_PRESENT));//从渲染目标到呈现
            screenRTState = D3D12_RESOURCE_STATE_PRESENT;
            m_CurrentBackBuffer = (m_CurrentBackBuffer + 1) % 2;
        }
    private:
        void CleanPreviousRenderTarget();

    private:
        // Share Objects (Basic + Writable)
        // ---------------------------------------
        // Device Objects
        ID3D12Device* m_d3dDevice;
        ID3D12GraphicsCommandList* ptr_CommandList;
        // RTV & DSV heaps
        ComPtr<ID3D12DescriptorHeap>    m_rtvHeap = nullptr;
        ComPtr<ID3D12DescriptorHeap>    m_dsvHeap = nullptr;
        // Size of all kinds of descriptor
        UINT  m_rtvDescriptorSize;
        UINT  m_dsvDescriptorSize;
        // Main Module
        MemoryManagerModule* MMModule;
        // Previous RT name
        std::string prevRTName = "NONE";

    private:
        // Pipeline Basic Objects
        // ---------------------------------------
        // Basic Screen RT & DS buffers
        IDXGISwapChain* m_SwapChain;
        ComPtr<ID3D12Resource>          m_SwapChainBuffer[2];
        ComPtr<ID3D12Resource>          m_DepthStencilBuffer;
        // Basic Rect & ViewPort
        D3D12_RECT scissorRect;
        D3D12_VIEWPORT viewPort;
        D3D12_RESOURCE_STATES screenRTState = D3D12_RESOURCE_STATE_PRESENT;
        // Basic Render Target Size
        float width, height;
        // Swap chain records
        UINT  m_CurrentBackBuffer = 0;

    public:

    private:
        // Writbale Addictive Objects
        // ---------------------------------------
        std::unordered_map<std::string, std::unique_ptr<WritableTexture>> mRTWritableTextures;
        std::unordered_map<std::string, std::unique_ptr<WritableTexture>> mDSWritableTextures;
        std::unordered_map<std::string, std::unique_ptr<WritableTexture>> mUAWritableTextures;
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

        ID3D12DescriptorHeap* CreateSRVHeap(std::string name, UINT num);

        void InitSRVHeap(RenderItemManager* manager);

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
        friend class RenderTargetTexture;

        MemoryManagerModule(ID3D12Device* device) :
            m_d3dDevice(device),
            RTDSSub(device, this),
            SRVSub(device, this)
        {}

        void GrabScreen()
        {
            RTDSSub.ReadBackToBuffer();
        }

        void EndNewFrame()
        {
            RTDSSub.EndNewFrame();
        }

        void SetSynchronizer(SynchronizationModule* sync)
        {
            synchronizer = sync;
        }

        void SetCommandList(ID3D12GraphicsCommandList* cmdList)
        {
            ptr_CommandList = cmdList;
            RTDSSub.ptr_CommandList = cmdList;
            SRVSub.ptr_CommandList = cmdList;
        }

        void InitSRVHeap(RenderItemManager* manager)
        {
            SRVSub.CreateSRVHeap("main", manager->mTextures.size());
            SRVSub.InitSRVHeap(manager);
        }

        RTDSSubmodule RTDSSub;
        ShaderResourceSubmodule SRVSub;

    private:
        ID3D12Device* m_d3dDevice;

        ID3D12GraphicsCommandList* ptr_CommandList;
        SynchronizationModule* synchronizer;
    };
}