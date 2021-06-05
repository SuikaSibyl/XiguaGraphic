#pragma once
#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <vector>
#include <d3d11_1.h>
#include <DirectXMath.h>

#include <assimp\Importer.hpp>
#include <assimp\scene.h>
#include <assimp\postprocess.h>

#include <MeshGeometry.h>
#include <GeometryGenerator.h>
#include <Texture.h>
#include <Geometry.h>

using namespace DirectX;

class QDirect3D12Widget;

class ModelLoader
{
	using uint16 = std::uint16_t;

public:
	ModelLoader(QDirect3D12Widget* widget, bool isLargeModel = false);
	~ModelLoader();

	void Load(std::string filename);
	bool LoadTransfer();
	std::unique_ptr<Geometry::MeshGeometry> GetMeshGeometry();
	Suika::CudaTriangleModel* GetCudaTriangle();
	void GetRenderItem();
	void Draw(ID3D11DeviceContext* devcon);
	std::vector<std::string> subname;
	int subnum = 0;
	bool isLargeModel = false;
	void Close();
	Geometry::MeshGeometryHelper helper;
	std::string name;

private:
	Geometry::MeshGeometry* meshGeo;

	//std::vector<Mesh> meshes;
	std::string directory;
	std::vector<Texture> textures_loaded;

	void processNode(aiNode* node, const aiScene* scene);
	void processMesh(aiMesh* mesh, const aiScene* scene);
	std::vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName, const aiScene* scene);
	std::string determineTextureType(const aiScene* scene, aiMaterial* mat);
	int getTextureIndex(aiString* str);
	ID3D11ShaderResourceView* getTextureFromModel(const aiScene* scene, int textureindex);
	QDirect3D12Widget* app = nullptr;
};

#endif // !MODEL_LOADER_H