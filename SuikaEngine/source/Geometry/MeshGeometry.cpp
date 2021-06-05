#include <MeshGeometry.h>
#include <QDirect3D12Widget.h>
#include "bvh.h"

std::unique_ptr<Geometry::MeshGeometry> Geometry::MeshGeometryHelper::CreateMeshGeometry(std::string name, bool isLargeModel)
{
	std::unique_ptr<MeshGeometry> geometry = std::make_unique<MeshGeometry>();
	geometry->Name = name;

	if (isLargeModel)
		geometry->IndexFormat = DXGI_FORMAT_R32_UINT;

	// Create all the submeshes
	UINT vertexOffset = 0;
	UINT indexOffset = 0;

	for (int i = 0; i < NameGroups.size(); i++)
	{
		Geometry::SubmeshGeometry submesh;
		// Write in the properties
		submesh.IndexCount = isLargeModel ? (UINT)IndicesGroups32[i].size() : (UINT)IndicesGroups16[i].size();
		submesh.BaseVertexLocation = vertexOffset;
		submesh.StartIndexLocation = indexOffset;
		// Set the submesh name
		geometry->DrawArgs[NameGroups[i]] = submesh;
		// Accumulate all the offset
		vertexOffset += (UINT)VerticesGroups[i].size();
		indexOffset += isLargeModel ? (UINT)IndicesGroups32[i].size() : (UINT)IndicesGroups16[i].size();
	}

	// Create the overall vertex vector
	std::vector<Geometry::Vertex> vertices;
	for (int i = 0; i < NameGroups.size(); i++)
		vertices.insert(vertices.end(), VerticesGroups[i].begin(), VerticesGroups[i].end());

	// Create the overall indices vector
	std::vector<std::uint16_t> indices16;
	std::vector<std::uint32_t> indices32;
	for (int i = 0; i < NameGroups.size(); i++)
	{
		if (isLargeModel)
			indices32.insert(indices32.end(), IndicesGroups32[i].begin(), IndicesGroups32[i].end());
		else
			indices16.insert(indices16.end(), IndicesGroups16[i].begin(), IndicesGroups16[i].end());
	}

	// Summarize the size of v&i
	const UINT vbByteSize = (UINT)vertices.size() * sizeof(Geometry::Vertex);
	const UINT ibByteSize = isLargeModel?(UINT)indices32.size() * sizeof(std::uint32_t): (UINT)indices16.size() * sizeof(std::uint16_t);

	geometry->VertexBufferGPU = ptr_d3dWidget->CreateDefaultBuffer(vbByteSize, vertices.data(), geometry->VertexBufferUploader);
	if (isLargeModel)
		geometry->IndexBufferGPU = ptr_d3dWidget->CreateDefaultBuffer(ibByteSize, indices32.data(), geometry->IndexBufferUploader);
	else
		geometry->IndexBufferGPU = ptr_d3dWidget->CreateDefaultBuffer(ibByteSize, indices16.data(), geometry->IndexBufferUploader);
	geometry->VertexByteStride = sizeof(Geometry::Vertex);
	geometry->VertexBufferByteSize = vbByteSize;
	geometry->IndexFormat = isLargeModel ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT;
	geometry->IndexBufferByteSize = ibByteSize;

	return std::move(geometry);
}

void MeshGeometryHelper::CalcNormal()
{
	for (int i = 0; i < NameGroups.size(); i++)
	{
		UINT mNumTriangles = IndicesGroups16[i].size() / 3;

		// For each triangle in the mesh:
		for (UINT j = 0; j < mNumTriangles; j++)
		{
			// indices of the ith triangle
			UINT i0 = IndicesGroups16[i][j * 3 + 0];
			UINT i1 = IndicesGroups16[i][j * 3 + 1];
			UINT i2 = IndicesGroups16[i][j * 3 + 2];

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

void MeshGeometryHelper::FillCudaModelIntermediate(Suika::CudaModelIntermediate& inter)
{
	// Create Vertex Array
	inter.g_vertices = GetCuVertices(inter.g_verticesNo);
	CuVertex* g_vertices = inter.g_vertices;
	unsigned& g_verticesNo = inter.g_verticesNo;

	// Create Triangle Array
	inter.g_triangles = GetCuTriangles(inter.g_trianglesNo, g_vertices);
	Triangle* g_triangles = inter.g_triangles;
	unsigned& g_trianglesNo = inter.g_trianglesNo;

	// Fix normals
	for (unsigned j = 0; j < g_trianglesNo; j++) {
		Vector3Df worldPointA = g_vertices[g_triangles[j]._idx1];
		Vector3Df worldPointB = g_vertices[g_triangles[j]._idx2];
		Vector3Df worldPointC = g_vertices[g_triangles[j]._idx3];
		Vector3Df AB = worldPointB;
		AB -= worldPointA;
		Vector3Df AC = worldPointC;
		AC -= worldPointA;
		Vector3Df cr = cross(AB, AC);
		cr.normalize();
		g_triangles[j]._normal = cr;
		g_vertices[g_triangles[j]._idx1]._normal += cr;
		g_vertices[g_triangles[j]._idx2]._normal += cr;
		g_vertices[g_triangles[j]._idx3]._normal += cr;
	}
	for (unsigned j = 0; j < g_trianglesNo; j++) {
		g_vertices[g_triangles[j]._idx1]._normal.normalize();
		g_vertices[g_triangles[j]._idx2]._normal.normalize();
		g_vertices[g_triangles[j]._idx3]._normal.normalize();
	}

	std::cout << "Vertices:  " << g_verticesNo << std::endl;
	std::cout << "Triangles: " << g_trianglesNo << std::endl;

	std::cout << "Pre-computing triangle intersection data (used by raytracer)..." << std::endl;

	for (unsigned i = 0; i < g_trianglesNo; i++) {

		Triangle& triangle = g_triangles[i];

		// Algorithm for triangle intersection is taken from Roman Kuchkuda's paper.
		// precompute edge vectors
		Vector3Df vc1 = g_vertices[triangle._idx2] - g_vertices[triangle._idx1];
		Vector3Df vc2 = g_vertices[triangle._idx3] - g_vertices[triangle._idx2];
		Vector3Df vc3 = g_vertices[triangle._idx1] - g_vertices[triangle._idx3];

		// plane of triangle, cross product of edge vectors vc1 and vc2
		triangle._normal = cross(vc1, vc2);

		// choose longest alternative normal for maximum precision
		Vector3Df alt1 = cross(vc2, vc3);
		if (alt1.length() > triangle._normal.length()) triangle._normal = alt1; // higher precision when triangle has sharp angles

		Vector3Df alt2 = cross(vc3, vc1);
		if (alt2.length() > triangle._normal.length()) triangle._normal = alt2;

		triangle._normal.normalize();

		// precompute dot product between normal and first triangle vertex
		triangle._d = dot(triangle._normal, g_vertices[triangle._idx1]);

		// edge planes
		triangle._e1 = cross(triangle._normal, vc1);
		triangle._e1.normalize();
		triangle._d1 = dot(triangle._e1, g_vertices[triangle._idx1]);
		triangle._e2 = cross(triangle._normal, vc2);
		triangle._e2.normalize();
		triangle._d2 = dot(triangle._e2, g_vertices[triangle._idx2]);
		triangle._e3 = cross(triangle._normal, vc3);
		triangle._e3.normalize();
		triangle._d3 = dot(triangle._e3, g_vertices[triangle._idx3]);
	}

	return;
}

Suika::CudaTriangleModel* MeshGeometryHelper::CreateCudaTriangle()
{
	Suika::CudaModelIntermediate intermediate;
	// Fill intermediate with g_vertices & g_triangles
	FillCudaModelIntermediate(intermediate);
	UpdateBoundingVolumeHierarchy(filename.c_str(), &intermediate.g_pSceneBVH, intermediate.g_triIndexListNo, &intermediate.h_TriIdxList, 
		intermediate.g_pCFBVH_No, &intermediate.g_pCFBVH, intermediate.g_vertices, intermediate.g_verticesNo,
		intermediate.g_triangles, intermediate.g_trianglesNo);
	intermediate.CreateBuffer();
	interm = std::move(intermediate);
	return interm.CreateCudaTriModel();
}