#include <RenderItemManagment.h>
#include <QDirect3D12Widget.h>


RenderItem* RenderItemManager::AddRitem(string geo_name, string sub_name, RenderQueue renderQ)
{
	std::unique_ptr<RenderItem> ritem = std::make_unique<RenderItem>();
	ritem->World = MathHelper::Identity4x4();
	ritem->ObjCBIndex = mAllRitems.size();
	ritem->Geo = geometries[geo_name].get();
	ritem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	ritem->IndexCount = geometries[geo_name]->DrawArgs[sub_name].IndexCount;
	ritem->BaseVertexLocation = geometries[geo_name]->DrawArgs[sub_name].BaseVertexLocation;
	ritem->StartIndexLocation = geometries[geo_name]->DrawArgs[sub_name].StartIndexLocation;

	switch (renderQ)
	{
	case Opaque:
		mOpaqueRitems.push_back(ritem.get());
		break;
	case Transparent:
		mTransparentRitems.push_back(ritem.get());
		break;
	default:
		break;
	}

	mAllRitems.push_back(std::move(ritem));

	return mAllRitems.back().get();
}

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

	return std::move(geometry);
}