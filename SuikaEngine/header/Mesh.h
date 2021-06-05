#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <bvh.h>
#include <string>

namespace Suika
{

	class CudaTriangleModel
	{
	public:
		int* d_TriIdxList;
		int* d_BVHIndex;
		float* d_BVHLimits;
		float* d_Triangles;

		~CudaTriangleModel();

		void BuildTriIndTexture(int* h_TriIdxList, unsigned h_TriIdxListNo);
		void BuildBVHLimitsTexture(float* h_BVHLimits, unsigned h_BVHLimitsNo);
		void BuildBVHIndexTexture(int* h_BVHIndex, unsigned h_BVHIndexNo);
		void BuildTriangleTexture(float* h_Triangles, unsigned h_TrianglesNo);

		CudaTriangleModel* GetDeviceVersion();

	private:

	};

	class CudaModelIntermediate
	{
	public:
		// The Model
		CuVertex* g_vertices = NULL;
		Triangle* g_triangles = NULL;

		// The BVH
		BVHNode* g_pSceneBVH = NULL;
		CacheFriendlyBVHNode* g_pCFBVH = NULL;

		// the cache-friendly version of the BVH
		unsigned	g_triIndexListNo = 0;
		unsigned	g_pCFBVH_No = 0;
		unsigned	g_verticesNo = 0;
		unsigned	g_trianglesNo = 0;

		float*	h_TriangleIntersectionData = NULL;
		int*	h_TriIdxList = NULL;
		float*	h_BVHlimits = NULL;
		int*	h_BVHindexesOrTrilists = NULL;

	public:
		void SetMesh(CuVertex* g_vertices, unsigned g_verticesNo,
			Triangle* g_triangles, unsigned g_trianglesNo)
		{
			this->g_vertices = g_vertices;
			this->g_verticesNo = g_verticesNo;
			this->g_triangles = g_triangles;
			this->g_trianglesNo = g_trianglesNo;
		}

		void CreateBuffer();
		CudaTriangleModel* CreateCudaTriModel();
	};

	class Mesh
	{

	};
}