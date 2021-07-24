#include <Precompiled.h>
#include "StructuredBuffer.h"

void StructuredBuffer::CreateUavDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE hCpuUav, CD3DX12_GPU_DESCRIPTOR_HANDLE hGpuUav)
{
	CPUHandle_UAV = hCpuUav;
	GPUHandle_UAV = hGpuUav;

	// Create DSV to resource
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavDesc.Format = DXGI_FORMAT_UNKNOWN;
	uavDesc.Buffer.FirstElement = 0;
	uavDesc.Buffer.NumElements = m_ElementNum;
	uavDesc.Buffer.StructureByteStride = m_PerEleSize;
	uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
	m_d3dDevice->CreateUnorderedAccessView(Resource.Get(), nullptr, &uavDesc, CPUHandle_UAV);
}


void StructuredBuffer::CreateSrvDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE hCpuSrv, CD3DX12_GPU_DESCRIPTOR_HANDLE hGpuSrv)
{
	CPUHandle_SRV = hCpuSrv;
	GPUHandle_SRV = hGpuSrv;

	// Create SRV to resource
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	m_d3dDevice->CreateShaderResourceView(Resource.Get(), &srvDesc, CPUHandle_SRV);
}

CD3DX12_GPU_DESCRIPTOR_HANDLE StructuredBuffer::gpuUav() const
{
	return GPUHandle_UAV;
}

CD3DX12_GPU_DESCRIPTOR_HANDLE StructuredBuffer::gpuSrv() const
{
	return GPUHandle_SRV;
}

D3D12_GPU_VIRTUAL_ADDRESS StructuredBuffer::gpuAddr() const
{
	return Resource.Get()->GetGPUVirtualAddress();
}