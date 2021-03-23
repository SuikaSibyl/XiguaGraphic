#include <RenderItemManagment.h>
#include <QDirect3D12Widget.h>

RenderItemManager::RenderItemManager(QDirect3D12Widget* ptr) :ptr_d3dWidget(ptr), helper(ptr) {};

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

	mQueueRitems[renderQ].push_back(ritem.get());

	mAllRitems.push_back(std::move(ritem));

	return mAllRitems.back().get();
}


void RenderItemManager::PushTexture(std::string name, std::wstring path, bool isCubemap)
{
	if (isCubemap)
	{
		mTextures[name] = helper.CreateCubemapTexture(name, path);
		mTextures[name]->isCubeMap = true;
	}
	else
		mTextures[name] = helper.CreateTexture(name, path);
	mTextures[name]->Index = mTextures.size() - 1;
}

void RenderItemManager::CreateTextureSRV()
{
	for (auto iter = mTextures.begin(); iter != mTextures.end(); iter++)
	{
		std::string name = iter->first;
		// Get pointer to the start of the heap.
		CD3DX12_CPU_DESCRIPTOR_HANDLE hDescriptor(ptr_d3dWidget->m_srvHeap->GetCPUDescriptorHandleForHeapStart());
		hDescriptor.Offset(mTextures[name]->Index, ptr_d3dWidget->m_cbv_srv_uavDescriptorSize);
		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Format = mTextures[name]->Resource->GetDesc().Format;
		srvDesc.ViewDimension = iter->second->isCubeMap ? D3D12_SRV_DIMENSION_TEXTURECUBE : D3D12_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MostDetailedMip = 0;
		srvDesc.Texture2D.MipLevels = mTextures[name]->Resource->GetDesc().MipLevels;
		srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
		ptr_d3dWidget->m_d3dDevice->CreateShaderResourceView(mTextures[name]->Resource.Get(), &srvDesc, hDescriptor);
	}
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

void MeshGeometryHelper::CalcNormal()
{
	for (int i = 0; i < NameGroups.size(); i++)
	{
		UINT mNumTriangles = IndicesGroups[i].size() / 3;

		// For each triangle in the mesh:
		for (UINT j = 0; j < mNumTriangles; j++)
		{
			// indices of the ith triangle
			UINT i0 = IndicesGroups[i][j * 3 + 0];
			UINT i1 = IndicesGroups[i][j * 3 + 1];
			UINT i2 = IndicesGroups[i][j * 3 + 2];

			// vertices of ith triangle
			Vertex v0 = VerticesGroups[i][i0];
			Vertex v1 = VerticesGroups[i][i1];
			Vertex v2 = VerticesGroups[i][i2];

			// compute face normal
			XMVECTOR e0 = XMLoadFloat3(&v1.Pos) - XMLoadFloat3(&v0.Pos);
			XMVECTOR e1 = XMLoadFloat3(&v2.Pos) - XMLoadFloat3(&v0.Pos);
			XMVECTOR faceNormal = XMVector3Cross(e0, e1);

			// This triangle shares the following three vertices, 
			// so add this face normal into the average of these
			// vertex normals. 
			XMStoreFloat3(&VerticesGroups[i][i0].Normal, faceNormal + XMLoadFloat3(&VerticesGroups[i][i0].Normal));
			XMStoreFloat3(&VerticesGroups[i][i1].Normal, faceNormal + XMLoadFloat3(&VerticesGroups[i][i1].Normal));
			XMStoreFloat3(&VerticesGroups[i][i2].Normal, faceNormal + XMLoadFloat3(&VerticesGroups[i][i2].Normal));
		}
		// For each vertex v, we have summed the face normals of all
		// the triangles that share v, so now we just need to normalize.
		for (UINT j = 0; j < VerticesGroups[i].size(); ++j)
		{
			XMVECTOR faceNormal = XMLoadFloat3(&VerticesGroups[i][j].Normal);
			faceNormal = XMVector3Normalize(faceNormal);
			XMStoreFloat3(&VerticesGroups[i][j].Normal, faceNormal);
		}
	}
}