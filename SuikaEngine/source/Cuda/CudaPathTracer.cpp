#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <iostream>
#include <math.h>
#include <CudaPathTracer.h>
#include <Options.h>

#ifndef M_PI
#define M_PI 3.14156265
#endif
#define width 1280	// screenwidth
#define height 720 // screenheight
using namespace std;

unsigned int framenumber = 0;

bool buffer_reset = false;

// image buffer storing accumulated pixel samples
Vector3Df* accumulatebuffer;
// final output buffer storing averaged pixel samples
Vector3Df* finaloutputbuffer;

// TODO: Delete stuff at some point!!!
Clock watch;
float scalefactor = 1.2f;

// this hash function calculates a new random number generator seed for each frame, based on framenumber  
unsigned int WangHash(unsigned int a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

// initialises scene data, builds BVH
void prepCUDAscene() {

	//// specify scene filename 
	//const char* scenefile = "../Resources/Models/dragon_vrip.ply";  // teapot.ply, big_atc.ply

	//// load scene
	//// Load scenefile to { g_vertices | g_triangles }
	//float maxi = load_object(scenefile);

	//// build the BVH
	//UpdateBoundingVolumeHierarchy(scenefile);

	//// now, allocate the CUDA side of the data (in CUDA global memory,
	//// in preparation for the textures that will store them...)

	//// store vertices in a GPU friendly format using float4
	//float* pVerticesData = (float*)malloc(g_verticesNo * 8 * sizeof(float));
	//for (unsigned f = 0; f < g_verticesNo; f++) {

	//	// first float4 stores vertex xyz position and precomputed ambient occlusion
	//	pVerticesData[f * 8 + 0] = g_vertices[f].x;
	//	pVerticesData[f * 8 + 1] = g_vertices[f].y;
	//	pVerticesData[f * 8 + 2] = g_vertices[f].z;
	//	pVerticesData[f * 8 + 3] = g_vertices[f]._ambientOcclusionCoeff;
	//	// second float4 stores vertex normal xyz
	//	pVerticesData[f * 8 + 4] = g_vertices[f]._normal.x;
	//	pVerticesData[f * 8 + 5] = g_vertices[f]._normal.y;
	//	pVerticesData[f * 8 + 6] = g_vertices[f]._normal.z;
	//	pVerticesData[f * 8 + 7] = 0.f;
	//}

	//// store precomputed triangle intersection data in a GPU friendly format using float4
	//float* pTrianglesIntersectionData = (float*)malloc(g_trianglesNo * 20 * sizeof(float));

	//for (unsigned e = 0; e < g_trianglesNo; e++) {
	//	// Texture-wise:
	//	//
	//	// first float4, triangle center + two-sided bool
	//	pTrianglesIntersectionData[20 * e + 0] = g_triangles[e]._center.x;
	//	pTrianglesIntersectionData[20 * e + 1] = g_triangles[e]._center.y;
	//	pTrianglesIntersectionData[20 * e + 2] = g_triangles[e]._center.z;
	//	pTrianglesIntersectionData[20 * e + 3] = g_triangles[e]._twoSided ? 1.0f : 0.0f;
	//	// second float4, normal
	//	pTrianglesIntersectionData[20 * e + 4] = g_triangles[e]._normal.x;
	//	pTrianglesIntersectionData[20 * e + 5] = g_triangles[e]._normal.y;
	//	pTrianglesIntersectionData[20 * e + 6] = g_triangles[e]._normal.z;
	//	pTrianglesIntersectionData[20 * e + 7] = g_triangles[e]._d;
	//	// third float4, precomputed plane normal of triangle edge 1
	//	pTrianglesIntersectionData[20 * e + 8] = g_triangles[e]._e1.x;
	//	pTrianglesIntersectionData[20 * e + 9] = g_triangles[e]._e1.y;
	//	pTrianglesIntersectionData[20 * e + 10] = g_triangles[e]._e1.z;
	//	pTrianglesIntersectionData[20 * e + 11] = g_triangles[e]._d1;
	//	// fourth float4, precomputed plane normal of triangle edge 2
	//	pTrianglesIntersectionData[20 * e + 12] = g_triangles[e]._e2.x;
	//	pTrianglesIntersectionData[20 * e + 13] = g_triangles[e]._e2.y;
	//	pTrianglesIntersectionData[20 * e + 14] = g_triangles[e]._e2.z;
	//	pTrianglesIntersectionData[20 * e + 15] = g_triangles[e]._d2;
	//	// fifth float4, precomputed plane normal of triangle edge 3
	//	pTrianglesIntersectionData[20 * e + 16] = g_triangles[e]._e3.x;
	//	pTrianglesIntersectionData[20 * e + 17] = g_triangles[e]._e3.y;
	//	pTrianglesIntersectionData[20 * e + 18] = g_triangles[e]._e3.z;
	//	pTrianglesIntersectionData[20 * e + 19] = g_triangles[e]._d3;
	//}

	//// copy precomputed triangle intersection data to CUDA global memory
	//cudaMalloc((void**)&cudaTriangleIntersectionData, g_trianglesNo * 20 * sizeof(float));
	//cudaMemcpy(cudaTriangleIntersectionData, pTrianglesIntersectionData, g_trianglesNo * 20 * sizeof(float), cudaMemcpyHostToDevice);

	//// Allocate CUDA-side data (global memory for corresponding textures) for Bounding Volume Hierarchy data
	//// See BVH.h for the data we are storing (from CacheFriendlyBVHNode)

	//// Leaf nodes triangle lists (indices to global triangle list)
	//// copy triangle indices to CUDA global memory
	//cudaMalloc((void**)&d_cudaTriIdxList, g_triIndexListNo * sizeof(int));
	//cudaMemcpy(d_cudaTriIdxList, g_triIndexList, g_triIndexListNo * sizeof(int), cudaMemcpyHostToDevice);

	//// Bounding box limits need bottom._x, top._x, bottom._y, top._y, bottom._z, top._z...
	//// store BVH bounding box limits in a GPU friendly format using float2
	//float* pLimits = (float*)malloc(g_pCFBVH_No * 6 * sizeof(float));

	//for (unsigned h = 0; h < g_pCFBVH_No; h++) {
	//	// Texture-wise:
	//	// First float2
	//	pLimits[6 * h + 0] = g_pCFBVH[h]._bottom.x;
	//	pLimits[6 * h + 1] = g_pCFBVH[h]._top.x;
	//	// Second float2
	//	pLimits[6 * h + 2] = g_pCFBVH[h]._bottom.y;
	//	pLimits[6 * h + 3] = g_pCFBVH[h]._top.y;
	//	// Third float2
	//	pLimits[6 * h + 4] = g_pCFBVH[h]._bottom.z;
	//	pLimits[6 * h + 5] = g_pCFBVH[h]._top.z;
	//}

	//// copy BVH limits to CUDA global memory
	//cudaMalloc((void**)&cudaBVHlimits, g_pCFBVH_No * 6 * sizeof(float));
	//cudaMemcpy(cudaBVHlimits, pLimits, g_pCFBVH_No * 6 * sizeof(float), cudaMemcpyHostToDevice);

	//// ..and finally, from CacheFriendlyBVHNode, the 4 integer values:
	//// store BVH node attributes (triangle count, startindex, left and right child indices) in a GPU friendly format using uint4
	//int* pIndexesOrTrilists = (int*)malloc(g_pCFBVH_No * 4 * sizeof(unsigned));

	//for (unsigned g = 0; g < g_pCFBVH_No; g++) {
	//	// Texture-wise:
	//	// A single uint4
	//	pIndexesOrTrilists[4 * g + 0] = g_pCFBVH[g].u.leaf._count;  // number of triangles stored in this node if leaf node
	//	pIndexesOrTrilists[4 * g + 1] = g_pCFBVH[g].u.inner._idxRight; // index to right child if inner node
	//	pIndexesOrTrilists[4 * g + 2] = g_pCFBVH[g].u.inner._idxLeft;  // index to left node if inner node
	//	pIndexesOrTrilists[4 * g + 3] = g_pCFBVH[g].u.leaf._startIndexInTriIndexList; // start index in list of triangle indices if leaf node
	//	// union

	//}

	//// copy BVH node attributes to CUDA global memory
	//cudaMalloc((void**)&cudaBVHindexesOrTrilists, g_pCFBVH_No * 4 * sizeof(unsigned));
	//cudaMemcpy(cudaBVHindexesOrTrilists, pIndexesOrTrilists, g_pCFBVH_No * 4 * sizeof(unsigned), cudaMemcpyHostToDevice);

	//// Initialisation Done!
	//std::cout << "Rendering data initialised and copied to CUDA global memory\n";
}

// display function called by glutMainLoop(), gets executed every frame 
void disp(void)
{
	// if camera has moved, reset the accumulation buffer
	if (buffer_reset) { cudaMemset(accumulatebuffer, 1, width * height * sizeof(Vector3Df)); framenumber = 0; }

	buffer_reset = false;
	framenumber++;

	cudaThreadSynchronize();

	// calculate a new seed for the random number generator, based on the framenumber
	unsigned int hashedframes = WangHash(framenumber);

	// gateway from host to CUDA, passes all data needed to render frame (triangles, BVH tree, camera) to CUDA for execution
	//cudarender(finaloutputbuffer, accumulatebuffer, cudaTriangles2, cudaBVHindexesOrTrilists2, cudaBVHlimits2, cudaTriangleIntersectionData2,
	//	cudaTriIdxList2, framenumber, hashedframes);

	cudaThreadSynchronize();
}

void PrepareCudaScene()
{
	// initialise all data needed to start rendering (BVH data, triangles, vertices)
	prepCUDAscene();
}

void CleanUp()
{
	system("PAUSE");
}
