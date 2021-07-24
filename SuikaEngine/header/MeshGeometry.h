#pragma once

#include <Utility.h>
#include <PipelineSetting.h>
#include <Platform/DirectX12/UploadBuffer.h>
#include <Material.h>
#include <Geometry.h>
#include <CudaPrt.h>
#include <Mesh.h>

class QDirect3D12Widget;

namespace Geometry
{
	// Define the vertex struct
	struct Vertex
	{
		XMFLOAT3 Pos;
		XMFLOAT3 Normal;
		XMFLOAT2 TexC;
		XMMATRIX SHTransfer;
		XMMATRIX DepthTransfer;
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

	class MeshGeometryHelper
	{
	public:
		MeshGeometryHelper(QDirect3D12Widget* d3dWidget)
		{
			ptr_d3dWidget = d3dWidget;
		}
		Suika::CudaModelIntermediate interm;
		void PushSubmeshGeometry(std::string name, std::vector<Vertex> vertices, std::vector<std::uint16_t> indices)
		{
			NameGroups.push_back(name);
			VerticesGroups.push_back(vertices);
			IndicesGroups16.push_back(indices);
		}

		void FillCudaModelIntermediate(Suika::CudaModelIntermediate& inter);

		void PushSubmeshGeometry(std::string name, std::vector<Vertex> vertices, std::vector<std::uint32_t> indices)
		{
			NameGroups.push_back(name);
			VerticesGroups.push_back(vertices);
			numVertex += vertices.size();
			IndicesGroups32.push_back(indices);
		}

		bool LoadPRT(std::string name)
		{
			PRTransfer ptr(numVertex);
			if (!ptr.LoadFromFile(name))
			{
				Debug::LogError(QString::fromStdString(name) + " not exist, please create transfer file first.");
				return false;
			}

			int index = 0;
			for (int i = 0; i < VerticesGroups.size(); i++)
			{
				for (int j = 0; j < VerticesGroups[i].size(); j++)
				{
					VerticesGroups[i][j].SHTransfer = XMMATRIX(
						ptr.pTransferData[index * 32 + 0], ptr.pTransferData[index * 32 + 1], ptr.pTransferData[index * 32 + 2], ptr.pTransferData[index * 32 + 3],
						ptr.pTransferData[index * 32 + 4], ptr.pTransferData[index * 32 + 5], ptr.pTransferData[index * 32 + 6], ptr.pTransferData[index * 32 + 7],
						ptr.pTransferData[index * 32 + 8], ptr.pTransferData[index * 32 + 9], ptr.pTransferData[index * 32 + 10], ptr.pTransferData[index * 32 + 11],
						ptr.pTransferData[index * 32 + 12], ptr.pTransferData[index * 32 + 13], ptr.pTransferData[index * 32 + 14], ptr.pTransferData[index * 32 + 15]);
					
					VerticesGroups[i][j].DepthTransfer = XMMATRIX(
						ptr.pTransferData[index * 32 + 0 + 16], ptr.pTransferData[index * 32 + 1 + 16], ptr.pTransferData[index * 32 + 2 + 16], ptr.pTransferData[index * 32 + 3 + 16],
						ptr.pTransferData[index * 32 + 4 + 16], ptr.pTransferData[index * 32 + 5 + 16], ptr.pTransferData[index * 32 + 6 + 16], ptr.pTransferData[index * 32 + 7 + 16],
						ptr.pTransferData[index * 32 + 8 + 16], ptr.pTransferData[index * 32 + 9 + 16], ptr.pTransferData[index * 32 + 10 + 16], ptr.pTransferData[index * 32 + 11 + 16],
						ptr.pTransferData[index * 32 + 12 + 16], ptr.pTransferData[index * 32 + 13 + 16], ptr.pTransferData[index * 32 + 14 + 16], ptr.pTransferData[index * 32 + 15 + 16]);
					index++;
				}
			}
			return true;
		}

		void CalcNormal();

		std::vector<Vertex>& GetVertices(int i = 0)
		{
			return VerticesGroups[i];
		};
		std::unique_ptr<MeshGeometry> CreateMeshGeometry(std::string name, bool isLargeModel = false);

		CuVertex* GetCuVertices(unsigned int & num)
		{
			for (int i = 0; i < VerticesGroups.size(); i++)
			{
				num += VerticesGroups[i].size();
			}
			CuVertex* cudaVertices = new CuVertex[num];
			int index = 0;
			for (int i = 0; i < VerticesGroups.size(); i++)
			{
				for (int j = 0; j < VerticesGroups[i].size(); j++)
				{
					cudaVertices[index].x = VerticesGroups[i][j].Pos.x * 10;
					cudaVertices[index].y = VerticesGroups[i][j].Pos.y * 10;
					cudaVertices[index].z = VerticesGroups[i][j].Pos.z * 10;

					cudaVertices[index]._normal.x = VerticesGroups[i][j].Normal.x;
					cudaVertices[index]._normal.y = VerticesGroups[i][j].Normal.y;
					cudaVertices[index]._normal.z = VerticesGroups[i][j].Normal.z;

					index++;
				}
			}
			return cudaVertices;
		}

		Triangle* GetCuTriangles(unsigned int& num, CuVertex* g_vertices)
		{
			float r, g, b;
			for (int i = 0; i < IndicesGroups32.size(); i++)
			{
				num += IndicesGroups32[i].size() / 3;
			}
			Triangle* cudaTriangles = new Triangle[num];
			Triangle* pCurrentTriangle = &(cudaTriangles[0]);

			for (int i = 0; i < IndicesGroups32.size(); i++)
			{
				for (int j = 0; j < IndicesGroups32[i].size(); j += 3)
				{
					// set rgb colour to white
					r = 255; g = 255; b = 255;

					pCurrentTriangle->_idx1 = IndicesGroups32[i][j];
					pCurrentTriangle->_idx2 = IndicesGroups32[i][j + 1];
					pCurrentTriangle->_idx3 = IndicesGroups32[i][j + 2];
					pCurrentTriangle->_colorf.x = r;
					pCurrentTriangle->_colorf.y = g;
					pCurrentTriangle->_colorf.z = b;
					pCurrentTriangle->_twoSided = false;
					pCurrentTriangle->_normal = Vector3Df(0, 0, 0);
					pCurrentTriangle->_bottom = Vector3Df(FLT_MAX, FLT_MAX, FLT_MAX);
					pCurrentTriangle->_top = Vector3Df(-FLT_MAX, -FLT_MAX, -FLT_MAX);
					CuVertex* vertexA = &g_vertices[IndicesGroups32[i][j]];
					CuVertex* vertexB = &g_vertices[IndicesGroups32[i][j + 1]];
					CuVertex* vertexC = &g_vertices[IndicesGroups32[i][j + 2]];
					pCurrentTriangle->_center = Vector3Df(
						(vertexA->x + vertexB->x + vertexC->x) / 3.0f,
						(vertexA->y + vertexB->y + vertexC->y) / 3.0f,
						(vertexA->z + vertexB->z + vertexC->z) / 3.0f);
					pCurrentTriangle++;
				}
			}
			return cudaTriangles;
		}

		std::string filename;
	private:
		QDirect3D12Widget* ptr_d3dWidget;

		int numVertex = 0;
		int numTriangle = 0;

		std::vector<std::string> NameGroups;
		std::vector<std::vector<Vertex>> VerticesGroups;
		std::vector<std::vector<std::uint16_t>> IndicesGroups16;
		std::vector<std::vector<std::uint32_t>> IndicesGroups32;

		bool isLargeModel = false;
		// Create Cuda Stuff
	public:
		Suika::CudaTriangleModel* CreateCudaTriangle();
	};
}