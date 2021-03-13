#pragma once

#include <vector>
#include <string>
#include <Utility.h>
#include <MeshGeometry.h>
#include <unordered_map>  
#include <map>

class QDirect3D12Widget;

using namespace Geometry;
using std::vector;
using std::string;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;

enum RenderQueue
{
	Opaque,
	Transparent,
};

class RenderItemManager
{
public:
	// List of all the render items.
	std::vector<std::unique_ptr<RenderItem>> mAllRitems;

	// Render items divided by PSO.
	std::vector<RenderItem*> mOpaqueRitems;
	std::vector<RenderItem*> mTransparentRitems;

	// Render items divided by PSO.
	std::unordered_map<std::string, std::unique_ptr<Geometry::MeshGeometry>> geometries;

	void AddGeometry(string name, std::unique_ptr<Geometry::MeshGeometry>& geo)
	{
		geometries[name] = std::move(geo);
	}

	RenderItem* AddRitem(string geo_name, string sub_name, RenderQueue renderQ = Opaque);

	void DisposeAllUploaders()
	{
		for (auto iter = geometries.begin(); iter != geometries.end(); iter++)
		{
			iter->second->DisposeUploaders();
		}
	}
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

	std::unique_ptr<MeshGeometry> CreateMeshGeometry(string name);

private:
	QDirect3D12Widget* ptr_d3dWidget;

	vector<string> NameGroups;
	vector<vector<Vertex>> VerticesGroups;
	vector<vector<std::uint16_t>> IndicesGroups;
};