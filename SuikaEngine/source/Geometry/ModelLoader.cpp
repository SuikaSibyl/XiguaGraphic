#include <Precompiled.h>
#include <ModelLoader.h>
#include <QDirect3D12Widget.h>
#include <CudaPrt.h>

ModelLoader::ModelLoader(QDirect3D12Widget* widget, bool isLargeModel) :helper(widget)
{
	this->isLargeModel = isLargeModel;
	app = widget;
}

ModelLoader::~ModelLoader()
{

}

Suika::CudaTriangleModel* ModelLoader::GetCudaTriangle()
{
	return helper.CreateCudaTriangle();
}

void ModelLoader::Load(std::string filename)
{
	Assimp::Importer importer;

	const aiScene* pScene = importer.ReadFile(filename,
		aiProcess_Triangulate |
		aiProcess_ConvertToLeftHanded);

	if (pScene == NULL)
		return;

	this->directory = filename.substr(0, filename.find_last_of('/'));
	std::string file = filename.substr(filename.find_last_of('/') + 1, filename.length() - filename.find_last_of('/') - 1);
	name = filename.substr(filename.find_last_of('/') + 1, filename.find_last_of('.') - filename.find_last_of('/') - 1);

	processNode(pScene->mRootNode, pScene);
	helper.filename = file;
	return;
}

bool ModelLoader::LoadTransfer()
{
	// Load success
	if (helper.LoadPRT(TRANSFER_PATH + name + ".transfer"))
		return true;
	// Load Failed
	else
		return false;
}

std::unique_ptr<Geometry::MeshGeometry> ModelLoader::GetMeshGeometry()
{
	return helper.CreateMeshGeometry(name, isLargeModel);
}

void ModelLoader::processNode(aiNode* node, const aiScene* scene)
{
	aiMatrix4x4 trans = node->mTransformation;
	for (UINT i = 0; i < node->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		this->processMesh(mesh, scene);
	}

	for (UINT i = 0; i < node->mNumChildren; i++)
	{
		this->processNode(node->mChildren[i], scene);
	}
}

void ModelLoader::Draw(ID3D11DeviceContext* devcon)
{

}

void ModelLoader::processMesh(aiMesh* mesh, const aiScene* scene)
{
	// Data to fill
	std::vector<Geometry::Vertex> vertices;
	std::vector<uint16> indices16;
	std::vector<uint32_t> indices32;
	std::vector<Texture> textures;

	// Walk through each of the mesh's vertices
	for (UINT i = 0; i < mesh->mNumVertices; i++)
	{
		Geometry::Vertex vertex;

		vertex.Pos = XMFLOAT3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);

		if (mesh->mNormals)
		{
			vertex.Normal = XMFLOAT3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
		}
		if (mesh->mTextureCoords[0])
		{
			vertex.TexC = XMFLOAT2((float)mesh->mTextureCoords[0][i].x, (float)mesh->mTextureCoords[0][i].y);
		}
		else
		{
			vertex.TexC = XMFLOAT2(0,0);
		}
		//vertex
		vertices.push_back(vertex);
	}

	for (UINT i = 0; i < mesh->mNumFaces; i++)
	{
		aiFace face = mesh->mFaces[i];

		for (UINT j = 0; j < face.mNumIndices; j++)
		{
			if (isLargeModel)
				indices32.push_back(face.mIndices[j]);
			else
				indices16.push_back(face.mIndices[j]);
		}
	}
	
	std::string name(mesh->mName.C_Str());
	name = name + std::to_string(subnum);
	subnum++;
	subname.push_back(name);
	if (isLargeModel)
		helper.PushSubmeshGeometry(name, vertices, indices32);
	else
		helper.PushSubmeshGeometry(name, vertices, indices16);
	return;
}