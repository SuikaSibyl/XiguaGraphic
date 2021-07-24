#pragma once
#include <Utility.h>

// ========================================================================================
// Structured Buffer
// ========================================================================================
struct LightSH
{
	float r, g, b;
	XMFLOAT4X4 SHCoeff_r;
	XMFLOAT4X4 SHCoeff_g;
	XMFLOAT4X4 SHCoeff_b;
};

typedef unsigned int uint;

class StructuredBuffer
{
public:
	StructuredBuffer(uint elementNum, uint perEleSize, ID3D12Device* d3dDevice)
		:m_d3dDevice(d3dDevice), m_ElementNum(elementNum), m_PerEleSize(perEleSize), Writable(true) {}

	StructuredBuffer(Microsoft::WRL::ComPtr<ID3D12Resource> resource, uint elementNum, uint perEleSize, ID3D12Device* d3dDevice)
		:m_d3dDevice(d3dDevice), Resource(resource), m_ElementNum(elementNum), m_PerEleSize(perEleSize), Writable(false) {}

	int UavIndex = 0;
	int SrvIndex = 0;

	uint m_ElementNum, m_PerEleSize;

	void CreateUavDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE hCpuUav, CD3DX12_GPU_DESCRIPTOR_HANDLE hGpuUav);
	void CreateSrvDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE hCpuSrv, CD3DX12_GPU_DESCRIPTOR_HANDLE hGpuSrv);

	CD3DX12_GPU_DESCRIPTOR_HANDLE gpuUav() const;
	CD3DX12_GPU_DESCRIPTOR_HANDLE gpuSrv() const;
	D3D12_GPU_VIRTUAL_ADDRESS gpuAddr() const;

	bool Writable = false;
protected:
	ID3D12Device* m_d3dDevice;

	Microsoft::WRL::ComPtr<ID3D12Resource> Resource = nullptr;
	CD3DX12_CPU_DESCRIPTOR_HANDLE CPUHandle_SRV;
	CD3DX12_CPU_DESCRIPTOR_HANDLE CPUHandle_UAV;
	CD3DX12_GPU_DESCRIPTOR_HANDLE GPUHandle_SRV;
	CD3DX12_GPU_DESCRIPTOR_HANDLE GPUHandle_UAV;
};

template <class T>
class SpecifiedStructuredBuffer :public StructuredBuffer
{
public:
	SpecifiedStructuredBuffer(uint num, ID3D12Device* d3dDevice);
private:
};

template <class T>
SpecifiedStructuredBuffer<T>::SpecifiedStructuredBuffer(uint itemNum, ID3D12Device* d3dDevice)
	:StructuredBuffer(itemNum, sizeof(T), d3dDevice)
{
	// Create Resource on Defualt Heap
	DXCall(d3dDevice->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(itemNum * sizeof(T), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		nullptr,
		IID_PPV_ARGS(&(this->Resource))));
}