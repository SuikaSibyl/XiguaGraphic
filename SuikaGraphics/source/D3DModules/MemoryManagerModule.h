#pragma once
#include <string>
#include <Utility.h>

using Microsoft::WRL::ComPtr;

class MemoryManagerModule
{
public:
	MemoryManagerModule(ID3D12Device* device) :m_d3dDevice(device) {}

	void CreateReadbackBuffer(UINT64 outputBufferSize)
	{
        // The output buffer (created below) is on a default heap, so only the GPU can access it.
        D3D12_HEAP_PROPERTIES defaultHeapProperties{ CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT) };
        D3D12_RESOURCE_DESC outputBufferDesc{ CD3DX12_RESOURCE_DESC::Buffer(outputBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) };
        ComPtr<ID3D12Resource> outputBuffer;
        DXCall(m_d3dDevice->CreateCommittedResource(
            &defaultHeapProperties,
            D3D12_HEAP_FLAG_NONE,
            &outputBufferDesc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(outputBuffer.GetAddressOf())));

        D3D12_HEAP_PROPERTIES readbackHeapProperties{ CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK) };
        D3D12_RESOURCE_DESC readbackBufferDesc{ CD3DX12_RESOURCE_DESC::Buffer(outputBufferSize) };
        ComPtr<ID3D12Resource> readbackBuffer;
        DXCall(m_d3dDevice->CreateCommittedResource(
            &readbackHeapProperties, //读回堆类型
            D3D12_HEAP_FLAG_NONE,
            &readbackBufferDesc, //默认堆为最终存储数据的地方，所以暂时初始化为普通状态
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(readbackBuffer.GetAddressOf())));

        {
            D3D12_RESOURCE_BARRIER outputBufferResourceBarrier
            {
                CD3DX12_RESOURCE_BARRIER::Transition(
                    outputBuffer.get(),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE)
            };
            commandList->ResourceBarrier(1, &outputBufferResourceBarrier);
        }
	}
private:
	ID3D12Device* m_d3dDevice;
};