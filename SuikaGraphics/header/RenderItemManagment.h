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
	std::unordered_map<std::string, std::unique_ptr<Geometry::MeshGeometry>> geometries;
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
		geometries[name] = std::move(geo);
	}

	void AddLight(string name, std::unique_ptr<Light>& light)
	{
		mLights[name] = std::move(light);
	}

	RenderItem* AddRitem(string geo_name, string sub_name, RenderQueue renderQ = Opaque);

	void DisposeAllUploaders()
	{
		for (auto iter = geometries.begin(); iter != geometries.end(); iter++)
		{
			iter->second->DisposeUploaders();
		}
	}

	void PushTexture(std::string name, std::wstring path, bool isCubemap = false);
	void CreateTextureSRV();
	void SetTexture(std::string mat_name, std::string texture_name)
	{
		mMaterials[mat_name]->DiffuseSrvHeapIndex = mTextures[texture_name]->Index;
	}

	void UpdateData();

private:
	QDirect3D12Widget* ptr_d3dWidget;
	TextureHelper helper;
};

class MeshGeometryHelper
{
public:
	MeshGeometryHelper(QDirect3D12Widget* d3dWidget)
	{
		ptr_d3dWidget = d3dWidget;
	}

	void PushSubmeshGeometry(string name, vector<Vertex> vertices, vector<std::uint16_t> indices)
	{
		NameGroups.push_back(name);
		VerticesGroups.push_back(vertices);
		IndicesGroups.push_back(indices);
	}

	void CalcNormal();
	vector<Vertex>& GetVertices(int i = 0)
	{
		return VerticesGroups[i];
	};
	std::unique_ptr<MeshGeometry> CreateMeshGeometry(string name);

private:
	QDirect3D12Widget* ptr_d3dWidget;

	vector<string> NameGroups;
	vector<vector<Vertex>> VerticesGroups;
	vector<vector<std::uint16_t>> IndicesGroups;
};