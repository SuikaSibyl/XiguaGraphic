#include <Precompiled.h>
#include <RenderItemManagment.h>
#include <QDirect3D12Widget.h>

RenderItemManager::RenderItemManager(QDirect3D12Widget* ptr) :ptr_d3dWidget(ptr), helper(ptr) {};

RenderItem* RenderItemManager::AddRitem(string geo_name, string sub_name, RenderQueue renderQ)
{
	std::unique_ptr<RenderItem> ritem = std::make_unique<RenderItem>();
	ritem->World = MathHelper::Identity4x4();
	ritem->ObjCBIndex = mAllRitems.size();
	ritem->Geo = mGeometries[geo_name].get();
	ritem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	ritem->IndexCount = mGeometries[geo_name]->DrawArgs[sub_name].IndexCount;
	ritem->BaseVertexLocation = mGeometries[geo_name]->DrawArgs[sub_name].BaseVertexLocation;
	ritem->StartIndexLocation = mGeometries[geo_name]->DrawArgs[sub_name].StartIndexLocation;

	mQueueRitems[renderQ].push_back(ritem.get());

	mAllRitems.push_back(std::move(ritem));

	return mAllRitems.back().get();
}


void RenderItemManager::PushTexture(std::string name, std::wstring path, Texture::Type type)
{
	switch (type)
	{
	case Texture::Texture2D:
		mTextures[name] = helper.CreateTexture(name, path);
		break;
	case Texture::Cubemap:
		mTextures[name] = helper.CreateCubemapTexture(name, path);
		mTextures[name]->isCubeMap = true;
		break;
	case Texture::CubemapArray:
		mTextures[name] = helper.CreateCubemapArray(name, path);
		break;
	default:
		break;
	}
	mTextures[name]->SrvIndex = mTextures.size() - 1;
}

void RenderItemManager::PushTextureCuda(std::string name, UINT width, UINT height, bool isCubemap)
{
	mTextures[name] = helper.CreateCudaTexture(name, width, height);
	mTextures[name]->SrvIndex = mTextures.size() - 1;
}

void RenderItemManager::CreateTextureSRV()
{
	// Create SRV Heap
	D3D12_DESCRIPTOR_HEAP_DESC srvDescriptorHeapDesc;
	srvDescriptorHeapDesc.NumDescriptors = mTextures.size();
	srvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	srvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	srvDescriptorHeapDesc.NodeMask = 0;
	ThrowIfFailed(ptr_d3dWidget->m_d3dDevice->CreateDescriptorHeap(&srvDescriptorHeapDesc, IID_PPV_ARGS(&(ptr_d3dWidget->m_srvHeap))));

	for (auto iter = mTextures.begin(); iter != mTextures.end(); iter++)
	{
		std::string name = iter->first;
		// Get pointer to the start of the heap.
		CD3DX12_CPU_DESCRIPTOR_HANDLE hDescriptor(ptr_d3dWidget->m_srvHeap->GetCPUDescriptorHandleForHeapStart());
		hDescriptor.Offset(mTextures[name]->SrvIndex, ptr_d3dWidget->m_cbv_srv_uavDescriptorSize);
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