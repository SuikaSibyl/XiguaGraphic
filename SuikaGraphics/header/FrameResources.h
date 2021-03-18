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

	//ÿ֡��Դ����Ҫ���������������
	ComPtr<ID3D12CommandAllocator> cmdAllocator;
	//ÿ֡����Ҫ��������Դ���������˰�����Ϊ2��������������
	std::unique_ptr<UploadBuffer<ObjectConstants>> objCB = nullptr;
	std::unique_ptr<UploadBuffer<PassConstants>> passCB = nullptr;
	std::unique_ptr<UploadBuffer<MaterialConstants>> materialCB = nullptr;
	std::unique_ptr<UploadBuffer<Geometry::Vertex>> dynamicVB = nullptr;
	std::unique_ptr<UploadBuffer<MaterialData>> materialSB = nullptr;
	//CPU�˵�Χ��ֵ
	UINT64 fenceCPU = 0;
};