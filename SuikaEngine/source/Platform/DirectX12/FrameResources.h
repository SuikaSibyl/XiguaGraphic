#pragma once
#include <utility.h>
#include <PipelineSetting.h>
#include "UploadBuffer.h"
#include <MeshGeometry.h>
#include <Material.h>
#include <Rendering/Lighting/PolygonLight.h>

struct FrameResource
{
public:
	FrameResource(ID3D12Device* device, UINT passCount, UINT objCount, UINT materialCount, UINT dynamicCount);
	FrameResource(const FrameResource& rhs) = delete;
	FrameResource& operator = (const FrameResource& rhs) = delete;
	~FrameResource();

	// Per frame Command Allocator
	ComPtr<ID3D12CommandAllocator> cmdAllocator;

	// Per frame buffers
	std::unique_ptr<UploadBuffer<ObjectConstants>> objCB = nullptr;
	std::unique_ptr<UploadBuffer<PassConstants>> passCB = nullptr;
	std::unique_ptr<UploadBuffer<Geometry::Vertex>> dynamicVB = nullptr;
	std::unique_ptr<UploadBuffer<MaterialData>> materialSB = nullptr;

	// CPU fence
	UINT64 fenceCPU = 0;
};