#include <Precompiled.h>
#include "FrameResources.h"

FrameResource::FrameResource(ID3D12Device* device, UINT passCount, UINT objCount, UINT materialCount, UINT dynamicCount)
{
	ThrowIfFailed(device->CreateCommandAllocator(
		D3D12_COMMAND_LIST_TYPE_DIRECT,
		IID_PPV_ARGS(&cmdAllocator)));

	objCB = std::make_unique<UploadBuffer<ObjectConstants>>(device, objCount, true);
	passCB = std::make_unique<UploadBuffer<PassConstants>>(device, passCount, true);
	dynamicVB = std::make_unique<UploadBuffer<Geometry::Vertex>>(device, dynamicCount, false);
	materialSB = std::make_unique<UploadBuffer<MaterialData>>(device, materialCount, false);
}

FrameResource::~FrameResource() {}