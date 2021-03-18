#pragma once
#include <utility.h>
#include <PipelineSetting.h>
#include <UploadBuffer.h>
#include <MeshGeometry.h>
#include <Material.h>

struct FrameResource
{
public:
	FrameResource(ID3D12Device* device, UINT passCount, UINT objCount, UINT materialCount, UINT dynamicCount);
	FrameResource(const FrameResource& rhs) = delete;
	FrameResource& operator = (const FrameResource& rhs) = delete;
	~FrameResource();

	//每帧资源都需要独立的命令分配器
	ComPtr<ID3D12CommandAllocator> cmdAllocator;
	//每帧都需要单独的资源缓冲区（此案例仅为2个常量缓冲区）
	std::unique_ptr<UploadBuffer<ObjectConstants>> objCB = nullptr;
	std::unique_ptr<UploadBuffer<PassConstants>> passCB = nullptr;
	std::unique_ptr<UploadBuffer<MaterialConstants>> materialCB = nullptr;
	std::unique_ptr<UploadBuffer<Geometry::Vertex>> dynamicVB = nullptr;
	std::unique_ptr<UploadBuffer<MaterialData>> materialSB = nullptr;
	//CPU端的围栏值
	UINT64 fenceCPU = 0;
};