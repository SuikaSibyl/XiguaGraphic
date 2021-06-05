#pragma once

#include <vector>
#include <string>
#include <Utility.h>
#include <MeshGeometry.h>
#include <unordered_map>  
#include <map>
#include <Material.h>
#include <Light.h>
#include <Texture.h>
#include <TextureHelper.h>
#include <Shader.h>

class QDirect3D12Widget;

using namespace Geometry;
using std::vector;
using std::string;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;

enum RenderQueue
{
	Opaque,
	AlphaTest,
	Transparent,
	Skybox,
	PostProcessing,
};

class RenderItemManager
{
public:
	RenderItemManager(QDirect3D12Widget* ptr);
	// List of all the render items.
	std::vector<std::unique_ptr<RenderItem>> mAllRitems;

	// Render items divided by PSO.
	std::unordered_map<RenderQueue, std::vector<RenderItem*>> mQueueRitems;

	// Render items divided by PSO.
	std::unordered_map<std::string, std::unique_ptr<Geometry::MeshGeometry>> mGeometries;
	// Materials
	std::unordered_map<std::string, std::unique_ptr<Material>> mMaterials;
	std::unordered_map<std::string, std::unique_ptr<Light>> mLights;
	std::unordered_map<std::string, std::unique_ptr<Texture>> mTextures;
	std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> mPSOs;
	std::unordered_map<std::string, std::unique_ptr<Shader>> mShaders;

	ComPtr<ID3D12PipelineState> AddPSO(string name)
	{
		ComPtr<ID3D12PipelineState> pso = nullptr;
		mPSOs[name] = pso;
		return mPSOs[name];
	}

	void AddGeometry(string name, std::unique_ptr<Geometry::MeshGeometry>& geo)
	{
		mGeometries[name] = std::move(geo);
	}

	void AddLight(string name, std::unique_ptr<Light>& light)
	{
		mLights[name] = std::move(light);
	}

	RenderItem* AddRitem(string geo_name, string sub_name, RenderQueue renderQ = Opaque);

	void DisposeAllUploaders()
	{
		for (auto iter = mGeometries.begin(); iter != mGeometries.end(); iter++)
		{
			iter->second->DisposeUploaders();
		}
		for (auto iter = mTextures.begin(); iter != mTextures.end(); iter++)
		{
			iter->second->DisposeUploaders();
		}
	}

	void PushTexture(std::string name, std::wstring path, Texture::Type type = Texture::Type::Texture2D);
	void PushTextureCuda(std::string name, UINT width, UINT height, bool isCubemap = false);

	void CreateTextureSRV();
	void SetTexture(std::string mat_name, std::string texture_name)
	{
		mMaterials[mat_name]->DiffuseSrvHeapIndex = mTextures[texture_name]->SrvIndex;
	}
	void SetTexture(std::string mat_name, int index)
	{
		mMaterials[mat_name]->DiffuseSrvHeapIndex = index;
	}

	void UpdateData();

private:
	QDirect3D12Widget* ptr_d3dWidget;
	TextureHelper helper;
};