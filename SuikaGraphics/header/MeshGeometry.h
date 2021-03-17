#pragma once

#include <Utility.h>
#include <PipelineSetting.h>
#include <memory>
#include <UploadBuffer.h>
#include <Material.h>

namespace Geometry
{
	// Define the vertex struct
	struct Vertex
	{
		XMFLOAT3 Pos;
		XMFLOAT3 Normal;
		XMFLOAT2 TexC;
	};

	// Defines a subrange of geometry in a MeshGeometry. This is for when
	// multiple geometries are stored in one vertex and index buffer.It
	// provides the offsets and data needed to draw a subset of geometry
	// stores in the vertex and index buffers so that we can implement the
	// technique described by Figure 6.3.
	struct SubmeshGeometry
	{
		UINT IndexCount = 0;
		UINT StartIndexLocation = 0;
		INT BaseVertexLocation = 0;
		// Bounding box of the geometry defined by this submesh.
		// This is used in later chapters of the book.
		DirectX::BoundingBox Bounds;
	};

	struct MeshGeometry;

	// Lightweight structure stores parameters to draw a shape.This will vary from app-to-app.
	struct RenderItem
	{
		RenderItem() = default;

		// World matrix of the shape that describes the object’s local space
		// relative to the world space, which defines the position,
		// orientation, and scale of the object in the world.
		XMFLOAT4X4 World = MathHelper::Identity4x4();
		//该几何体的顶点UV缩放矩阵
		XMFLOAT4X4 texTransform = MathHelper::Identity4x4();

		// Dirty flag indicating the object data has changed and we need
		// to update the constant buffer. Because we have an object
		// cbuffer for each FrameResource, we have to apply the
		// update to each FrameResource. Thus, when we modify obect data we
		// should set
		// NumFramesDirty = gNumFrameResources so that each frame resource
		// gets the update.
		int NumFramesDirty = frameResourcesCount;

		// Index into GPU constant buffer corresponding to the ObjectCB
		// for this render item.
		UINT ObjCBIndex = -1;

		// Geometry associated with this render-item. Note that multiple
		// render-items can share the same geometry.
		MeshGeometry* Geo = nullptr;
		// Material
		Material* material = nullptr;

		// Geometry associated with this render-item. Note that multiple
		// render-items can share the same geometry.MeshGeometry* Geo = nullptr;
		// Primitive topology.
		D3D12_PRIMITIVE_TOPOLOGY PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

		// DrawIndexedInstanced parameters.
		UINT IndexCount = 0;
		UINT StartIndexLocation = 0;
		int BaseVertexLocation = 0;
	};

	struct MeshGeometry
	{
		// Give it a name so we can look it up by name.
		std::string Name;
		// System memory copies. Use Blobs because the vertex / index format can
		// be generic.
		// It is up to the client to cast appropriately.
		Microsoft::WRL::ComPtr<ID3DBlob> VertexBufferCPU = nullptr;
		Microsoft::WRL::ComPtr<ID3DBlob> IndexBufferCPU = nullptr;

		Microsoft::WRL::ComPtr<ID3D12Resource> VertexBufferGPU = nullptr;
		Microsoft::WRL::ComPtr<ID3D12Resource> IndexBufferGPU = nullptr;

		Microsoft::WRL::ComPtr<ID3D12Resource> VertexBufferUploader = nullptr;
		Microsoft::WRL::ComPtr<ID3D12Resource> IndexBufferUploader = nullptr;

		// Data about the buffers.
		UINT VertexByteStride = 0;
		UINT VertexBufferByteSize = 0;
		DXGI_FORMAT IndexFormat = DXGI_FORMAT_R16_UINT;
		UINT IndexBufferByteSize = 0;

		// A MeshGeometry may store multiple geometries in one vertex / index buffer.
		// Use this container to define the Submesh geometries so we can draw the Submeshes individually.
		std::unordered_map<std::string, SubmeshGeometry> DrawArgs;
		std::vector<std::unique_ptr<RenderItem>> RenderItems;

		D3D12_VERTEX_BUFFER_VIEW VertexBufferView()const
		{
			D3D12_VERTEX_BUFFER_VIEW vbv;
			vbv.BufferLocation = VertexBufferGPU->GetGPUVirtualAddress();
			vbv.StrideInBytes = VertexByteStride;
			vbv.SizeInBytes = VertexBufferByteSize;
			return vbv;
		}
		D3D12_INDEX_BUFFER_VIEW IndexBufferView()const
		{
			D3D12_INDEX_BUFFER_VIEW ibv;
			ibv.BufferLocation = IndexBufferGPU->GetGPUVirtualAddress();
			ibv.Format = IndexFormat;
			ibv.SizeInBytes = IndexBufferByteSize;
			return ibv;
		}
		// We can free this memory after we finish upload to the GPU.
		void DisposeUploaders()
		{
			VertexBufferUploader = nullptr;
			IndexBufferUploader = nullptr;
		}
	};
}