#include <RenderItemManagment.h>
#include <QDirect3D12Widget.h>

std::unique_ptr<MeshGeometry> MeshGeometryHelper::CreateMeshGeometry(string name)
{
	std::unique_ptr<MeshGeometry> geometry = std::make_unique<MeshGeometry>();
	geometry->Name = name;

	// Create all the submeshes
	UINT vertexOffset = 0;
	UINT indexOffset = 0;

	for (int i = 0; i < NameGroups.size(); i++)
	{
		Geometry::SubmeshGeometry submesh;
		// Write in the properties
		submesh.IndexCount = (UINT)IndicesGroups[i].size();
		submesh.BaseVertexLocation = vertexOffset;
		submesh.StartIndexLocation = indexOffset;
		// Set the submesh name
		geometry->DrawArgs[NameGroups[i]] = submesh;
		// Accumulate all the offset
		vertexOffset += (UINT)VerticesGroups[i].size();
		indexOffset += (UINT)IndicesGroups[i].size();
	}

	// Create the overall vertex vector
	std::vector<Geometry::Vertex> vertices;
	for (int i = 0; i < NameGroups.size(); i++)
		vertices.insert(vertices.end(), VerticesGroups[i].begin(), VerticesGroups[i].end());

	// Create the overall indices vector
	std::vector<std::uint16_t> indices;
	for (int i = 0; i < NameGroups.size(); i++)
		indices.insert(indices.end(), IndicesGroups[i].begin(), IndicesGroups[i].end());

	// Summarize the size of v&i
	const UINT vbByteSize = (UINT)vertices.size() * sizeof(Geometry::Vertex);
	const UINT ibByteSize = (UINT)indices.size() * sizeof(std::uint16_t);

	geometry->VertexBufferGPU = ptr_d3dWidget->CreateDefaultBuffer(vbByteSize, vertices.data(), geometry->VertexBufferUploader);
	geometry->IndexBufferGPU = ptr_d3dWidget->CreateDefaultBuffer(ibByteSize, indices.data(), geometry->IndexBufferUploader);
	geometry->VertexByteStride = sizeof(Geometry::Vertex);
	geometry->VertexBufferByteSize = vbByteSize;
	geometry->IndexFormat = DXGI_FORMAT_R16_UINT;
	geometry->IndexBufferByteSize = ibByteSize;

	//auto gridRitem = std::make_unique<RenderItem>();
	//gridRitem->World = MathHelper::Identity4x4();
	//gridRitem->ObjCBIndex = 0;
	//gridRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	//gridRitem->IndexCount = mMultiGeo->DrawArgs["grid"].IndexCount;
	//gridRitem->BaseVertexLocation = mMultiGeo->DrawArgs["grid"].BaseVertexLocation;
	//gridRitem->StartIndexLocation = mMultiGeo->DrawArgs["grid"].StartIndexLocation;
	//mMultiGeo->RenderItems.push_back(std::move(gridRitem));

	return std::move(geometry);
}