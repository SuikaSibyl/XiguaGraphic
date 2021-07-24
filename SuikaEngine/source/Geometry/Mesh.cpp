#include <Precompiled.h>
#include <Mesh.h>
#include <CudaUtil.h>
#include <CudaPathTracer.h>

using namespace Suika;

void CudaTriangleModel::BuildTriIndTexture(int* h_TriIdxList, unsigned h_TriIdxListNo)
{
	checkCudaErrors(cudaMalloc((void**)&d_TriIdxList, h_TriIdxListNo * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_TriIdxList, h_TriIdxList, h_TriIdxListNo * sizeof(int), cudaMemcpyHostToDevice));
}

void CudaTriangleModel::BuildBVHLimitsTexture(float* h_BVHLimits, unsigned h_BVHLimitsNo)
{
	checkCudaErrors(cudaMalloc((void**)&d_BVHLimits, h_BVHLimitsNo * 6 * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_BVHLimits, h_BVHLimits, h_BVHLimitsNo * 6 * sizeof(float), cudaMemcpyHostToDevice));
}

void CudaTriangleModel::BuildBVHIndexTexture(int* h_BVHIndex, unsigned h_BVHIndexNo)
{
	checkCudaErrors(cudaMalloc((void**)&d_BVHIndex, h_BVHIndexNo * 4 * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_BVHIndex, h_BVHIndex, h_BVHIndexNo * 4 * sizeof(int), cudaMemcpyHostToDevice));
}

void CudaTriangleModel::BuildTriangleTexture(float* h_Triangles, unsigned h_TrianglesNo)
{
	checkCudaErrors(cudaMalloc((void**)&d_Triangles, h_TrianglesNo * 20 * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_Triangles, h_Triangles, h_TrianglesNo * 20 * sizeof(float), cudaMemcpyHostToDevice));
}

CudaTriangleModel* CudaTriangleModel::GetDeviceVersion()
{
	Suika::CudaTriangleModel* d_model = nullptr;
	cudaMalloc((void**)&d_model, 1 * sizeof(Suika::CudaTriangleModel));
	cudaMemcpy(&(d_model->d_TriIdxList), &(d_TriIdxList), sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_model->d_BVHLimits), &(d_BVHLimits), sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_model->d_BVHIndex), &(d_BVHIndex), sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_model->d_Triangles), &(d_Triangles), sizeof(float*), cudaMemcpyHostToDevice);
	return d_model;
}

CudaTriangleModel::~CudaTriangleModel()
{
	// Free All device Memory
	//cudaFree(d_TriIdxList);
	//cudaFree(d_BVHIndex);
	//cudaFree(d_BVHLimits);
	//cudaFree(d_Triangles);
}

void CudaModelIntermediate::CreateBuffer()
{
	// now, allocate the CUDA side of the data (in CUDA global memory,
	// in preparation for the textures that will store them...)

	// store vertices in a GPU friendly format using float4
	float* pVerticesData = (float*)malloc(g_verticesNo * 8 * sizeof(float));
	for (unsigned f = 0; f < g_verticesNo; f++) {

		// first float4 stores vertex xyz position and precomputed ambient occlusion
		pVerticesData[f * 8 + 0] = g_vertices[f].x;
		pVerticesData[f * 8 + 1] = g_vertices[f].y;
		pVerticesData[f * 8 + 2] = g_vertices[f].z;
		pVerticesData[f * 8 + 3] = g_vertices[f]._ambientOcclusionCoeff;
		// second float4 stores vertex normal xyz
		pVerticesData[f * 8 + 4] = g_vertices[f]._normal.x;
		pVerticesData[f * 8 + 5] = g_vertices[f]._normal.y;
		pVerticesData[f * 8 + 6] = g_vertices[f]._normal.z;
		pVerticesData[f * 8 + 7] = 0.f;
	}

	// store precomputed triangle intersection data in a GPU friendly format using float4
	float* pTrianglesIntersectionData = (float*)malloc(g_trianglesNo * 20 * sizeof(float));

	for (unsigned e = 0; e < g_trianglesNo; e++) {
		// Texture-wise:
		//
		// first float4, triangle center + two-sided bool
		pTrianglesIntersectionData[20 * e + 0] = g_triangles[e]._center.x;
		pTrianglesIntersectionData[20 * e + 1] = g_triangles[e]._center.y;
		pTrianglesIntersectionData[20 * e + 2] = g_triangles[e]._center.z;
		pTrianglesIntersectionData[20 * e + 3] = g_triangles[e]._twoSided ? 1.0f : 0.0f;
		// second float4, normal
		pTrianglesIntersectionData[20 * e + 4] = g_triangles[e]._normal.x;
		pTrianglesIntersectionData[20 * e + 5] = g_triangles[e]._normal.y;
		pTrianglesIntersectionData[20 * e + 6] = g_triangles[e]._normal.z;
		pTrianglesIntersectionData[20 * e + 7] = g_triangles[e]._d;
		// third float4, precomputed plane normal of triangle edge 1
		pTrianglesIntersectionData[20 * e + 8] = g_triangles[e]._e1.x;
		pTrianglesIntersectionData[20 * e + 9] = g_triangles[e]._e1.y;
		pTrianglesIntersectionData[20 * e + 10] = g_triangles[e]._e1.z;
		pTrianglesIntersectionData[20 * e + 11] = g_triangles[e]._d1;
		// fourth float4, precomputed plane normal of triangle edge 2
		pTrianglesIntersectionData[20 * e + 12] = g_triangles[e]._e2.x;
		pTrianglesIntersectionData[20 * e + 13] = g_triangles[e]._e2.y;
		pTrianglesIntersectionData[20 * e + 14] = g_triangles[e]._e2.z;
		pTrianglesIntersectionData[20 * e + 15] = g_triangles[e]._d2;
		// fifth float4, precomputed plane normal of triangle edge 3
		pTrianglesIntersectionData[20 * e + 16] = g_triangles[e]._e3.x;
		pTrianglesIntersectionData[20 * e + 17] = g_triangles[e]._e3.y;
		pTrianglesIntersectionData[20 * e + 18] = g_triangles[e]._e3.z;
		pTrianglesIntersectionData[20 * e + 19] = g_triangles[e]._d3;
	}

	// copy precomputed triangle intersection data to CUDA global memory
	h_TriangleIntersectionData = pTrianglesIntersectionData;

	// Bounding box limits need bottom._x, top._x, bottom._y, top._y, bottom._z, top._z...
	// store BVH bounding box limits in a GPU friendly format using float2
	float* pLimits = (float*)malloc(g_pCFBVH_No * 6 * sizeof(float));

	for (unsigned h = 0; h < g_pCFBVH_No; h++) {
		// Texture-wise:
		// First float2
		pLimits[6 * h + 0] = g_pCFBVH[h]._bottom.x;
		pLimits[6 * h + 1] = g_pCFBVH[h]._top.x;
		// Second float2
		pLimits[6 * h + 2] = g_pCFBVH[h]._bottom.y;
		pLimits[6 * h + 3] = g_pCFBVH[h]._top.y;
		// Third float2
		pLimits[6 * h + 4] = g_pCFBVH[h]._bottom.z;
		pLimits[6 * h + 5] = g_pCFBVH[h]._top.z;
	}

	h_BVHlimits = pLimits;

	// ..and finally, from CacheFriendlyBVHNode, the 4 integer values:
	// store BVH node attributes (triangle count, startindex, left and right child indices) in a GPU friendly format using uint4
	int* pIndexesOrTrilists = (int*)malloc(g_pCFBVH_No * 4 * sizeof(unsigned));

	for (unsigned g = 0; g < g_pCFBVH_No; g++) {
		// Texture-wise:
		// A single uint4
		pIndexesOrTrilists[4 * g + 0] = g_pCFBVH[g].u.leaf._count;  // number of triangles stored in this node if leaf node
		pIndexesOrTrilists[4 * g + 1] = g_pCFBVH[g].u.inner._idxRight; // index to right child if inner node
		pIndexesOrTrilists[4 * g + 2] = g_pCFBVH[g].u.inner._idxLeft;  // index to left node if inner node
		pIndexesOrTrilists[4 * g + 3] = g_pCFBVH[g].u.leaf._startIndexInTriIndexList; // start index in list of triangle indices if leaf node
		// union

	}

	h_BVHindexesOrTrilists = pIndexesOrTrilists;

	// Initialisation Done!
	std::cout << "Rendering data initialised and copied to CUDA global memory\n";
}

CudaTriangleModel* CudaModelIntermediate::CreateCudaTriModel()
{
	CudaTriangleModel* model = new CudaTriangleModel();
	model->BuildTriIndTexture(h_TriIdxList, g_triIndexListNo);
	model->BuildBVHLimitsTexture(h_BVHlimits, g_pCFBVH_No);
	model->BuildBVHIndexTexture(h_BVHindexesOrTrilists, g_pCFBVH_No);
	model->BuildTriangleTexture(h_TriangleIntersectionData, g_trianglesNo);
	return model;
}