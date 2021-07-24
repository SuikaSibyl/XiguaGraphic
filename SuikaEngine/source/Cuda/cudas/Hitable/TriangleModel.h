#pragma once
#ifndef TRIANGLEMODEL
#define TRIANGLEMODEL

#include "RTUtils.cuh"
#include <CudaPathTracer.h>
#include <linear_algebra.h>
#include <Mesh.h>

#define NUDGE_FACTOR     1e-3f  // epsilon

__device__ float4 FetchFloat4(float* buffer, int index)
{
	return make_float4(buffer[4 * index], buffer[4 * index + 1], buffer[4 * index + 2], buffer[4 * index + 3]);
}

// Helper function, that checks whether a ray intersects a bounding box (BVH node)
__device__ bool RayIntersectsBox(const Vector3Df& originInWorldSpace, const Vector3Df& rayInWorldSpace, int boxIdx, Suika::CudaTriangleModel* model)
{
	// set Tnear = - infinity, Tfar = infinity
	//
	// For each pair of planes P associated with X, Y, and Z do:
	//     (example using X planes)
	//     if direction Xd = 0 then the ray is parallel to the X planes, so
	//         if origin Xo is not between the slabs ( Xo < Xl or Xo > Xh) then
	//             return false
	//     else, if the ray is not parallel to the plane then
	//     begin
	//         compute the intersection distance of the planes
	//         T1 = (Xl - Xo) / Xd
	//         T2 = (Xh - Xo) / Xd
	//         If T1 > T2 swap (T1, T2) /* since T1 intersection with near plane */
	//         If T1 > Tnear set Tnear =T1 /* want largest Tnear */
	//         If T2 < Tfar set Tfar="T2" /* want smallest Tfar */
	//         If Tnear > Tfar box is missed so
	//             return false
	//         If Tfar < 0 box is behind ray
	//             return false
	//     end
	// end of for loop

	float Tnear, Tfar;
	Tnear = -FLT_MAX;
	Tfar = FLT_MAX;

	float2 limits;

	// box intersection routine
#define CHECK_NEAR_AND_FAR_INTERSECTION(c)							    \
    if (rayInWorldSpace.##c == 0.f) {						    \
	if (originInWorldSpace.##c < limits.x) return false;					    \
	if (originInWorldSpace.##c > limits.y) return false;					    \
	} else {											    \
	float T1 = (limits.x - originInWorldSpace.##c)/rayInWorldSpace.##c;			    \
	float T2 = (limits.y - originInWorldSpace.##c)/rayInWorldSpace.##c;			    \
	if (T1>T2) { float tmp=T1; T1=T2; T2=tmp; }						    \
	if (T1 > Tnear) Tnear = T1;								    \
	if (T2 < Tfar)  Tfar = T2;								    \
	if (Tnear > Tfar)	return false;									    \
	if (Tfar < 0.f)	return false;									    \
	}

	// box.bottom._x/top._x placed in limits.x/limits.y
	limits = make_float2(model->d_BVHLimits[6 * boxIdx + 0], model->d_BVHLimits[6 * boxIdx + 1]);
	CHECK_NEAR_AND_FAR_INTERSECTION(x)
		// box.bottom._y/top._y placed in limits.x/limits.y
		limits = make_float2(model->d_BVHLimits[6 * boxIdx + 2], model->d_BVHLimits[6 * boxIdx + 3]);
	CHECK_NEAR_AND_FAR_INTERSECTION(y)
		// box.bottom._z/top._z placed in limits.x/limits.y
		limits = make_float2(model->d_BVHLimits[6 * boxIdx + 4], model->d_BVHLimits[6 * boxIdx + 5]);
	CHECK_NEAR_AND_FAR_INTERSECTION(z)

		// If Box survived all above tests, return true with intersection point Tnear and exit point Tfar.
		return true;
}


//////////////////////////////////////////
//	BVH intersection routine	//
//	using CUDA texture memory	//
//////////////////////////////////////////

// there are 3 forms of the BVH: a "pure" BVH, a cache-friendly BVH (taking up less memory space than the pure BVH)
// and a "textured" BVH which stores its data in CUDA texture memory (which is cached). The last one is gives the 
// best performance and is used here.

__device__ bool BVH_IntersectTriangles(
	int* cudaBVHindexesOrTrilists, const Vector3Df& origin, const Vector3Df& ray, unsigned avoidSelf,
	int& pBestTriIdx, Vector3Df& pointHitInWorldSpace, float& kAB, float& kBC, float& kCA, float& hitdist,
	float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* d_cudaTriIdxList, Vector3Df& boxnormal,
	Suika::CudaTriangleModel* mesh)
{
	// in the loop below, maintain the closest triangle and the point where we hit it:
	pBestTriIdx = -1;
	float bestTriDist;

	// start from infinity
	bestTriDist = FLT_MAX;

	// create a stack for each ray
	// the stack is just a fixed size array of indices to BVH nodes
	int stack[BVH_STACK_SIZE];

	int stackIdx = 0;
	stack[stackIdx++] = 0;
	Vector3Df hitpoint;

	// while the stack is not empty
	while (stackIdx) {

		// pop a BVH node (or AABB, Axis Aligned Bounding Box) from the stack
		int boxIdx = stack[stackIdx - 1];
		//uint* pCurrent = &cudaBVHindexesOrTrilists[boxIdx]; 

		// decrement the stackindex
		stackIdx--;

		// fetch the data (indices to childnodes or index in triangle list + trianglecount) associated with this node
		uint4 data = make_uint4(mesh->d_BVHIndex[4 * boxIdx], mesh->d_BVHIndex[4 * boxIdx + 1],
			mesh->d_BVHIndex[4 * boxIdx + 2], mesh->d_BVHIndex[4 * boxIdx + 3]);

		// original, "pure" BVH form...
		//if (!pCurrent->IsLeaf()) {

		// cache-friendly BVH form...
		//if (!(cudaBVHindexesOrTrilists[4 * boxIdx + 0] & 0x80000000)) { // INNER NODE

		// texture memory BVH form...

		// determine if BVH node is an inner node or a leaf node by checking the highest bit (bitwise AND operation)
		// inner node if highest bit is 1, leaf node if 0

		if (!(data.x & 0x80000000)) {   // INNER NODE

			// if ray intersects inner node, push indices of left and right child nodes on the stack
			if (RayIntersectsBox(origin, ray, boxIdx, mesh)) {

				//stack[stackIdx++] = pCurrent->u.inner._idxRight;
				//stack[stackIdx++] = cudaBVHindexesOrTrilists[4 * boxIdx + 1];
				stack[stackIdx++] = data.y; // right child node index

				//stack[stackIdx++] = pCurrent->u.inner._idxLeft;
				//stack[stackIdx++] = cudaBVHindexesOrTrilists[4 * boxIdx + 2];
				stack[stackIdx++] = data.z; // left child node index

				// return if stack size is exceeded
				if (stackIdx > BVH_STACK_SIZE)
				{
					return false;
				}
			}
		}
		else { // LEAF NODE

			// original, "pure" BVH form...
			// BVHLeaf *p = dynamic_cast<BVHLeaf*>(pCurrent);
			// for(std::list<const Triangle*>::iterator it=p->_triangles.begin();
			//    it != p->_triangles.end();
			//    it++)

			// cache-friendly BVH form...
			// for(unsigned i=pCurrent->u.leaf._startIndexInTriIndexList;
			//    i<pCurrent->u.leaf._startIndexInTriIndexList + (pCurrent->u.leaf._count & 0x7fffffff);

			// texture memory BVH form...
			// for (unsigned i = cudaBVHindexesOrTrilists[4 * boxIdx + 3]; i< cudaBVHindexesOrTrilists[4 * boxIdx + 3] + (cudaBVHindexesOrTrilists[4 * boxIdx + 0] & 0x7fffffff); i++) { // data.w = number of triangles in leaf

			// loop over every triangle in the leaf node
			// data.w is start index in triangle list
			// data.x stores number of triangles in leafnode (the bitwise AND operation extracts the triangle number)
			for (unsigned i = data.w; i < data.w + (data.x & 0x7fffffff); i++) {

				// original, "pure" BVH form...
				//const Triangle& triangle = *(*it);

				// cache-friendly BVH form...
				//const Triangle& triangle = pTriangles[cudaTriIdxList[i]];

				// texture memory BVH form...
				// fetch the index of the current triangle
				// ORIGIN //int idx = tex1Dfetch(g_triIdxListTexture, i).x;
				int idx = mesh->d_TriIdxList[i];
				//int idx = cudaTriIdxList[i];

				// check if triangle is the same as the one intersected by previous ray
				// to avoid self-reflections/refractions
				if (avoidSelf == idx)
					continue;

				// fetch triangle center and normal from texture memory
				float4 center = FetchFloat4(mesh->d_Triangles, 5 * idx);
				float4 normal = FetchFloat4(mesh->d_Triangles, 5 * idx + 1);

				// use the pre-computed triangle intersection data: normal, d, e1/d1, e2/d2, e3/d3
				float k = dot(normal, ray);
				if (k == 0.0f)
					continue; // this triangle is parallel to the ray, ignore it.

				float s = (normal.w - dot(normal, origin)) / k;
				if (s <= 0.0f) // this triangle is "behind" the origin.
					continue;
				if (s <= NUDGE_FACTOR)  // epsilon
					continue;
				Vector3Df hit = ray * s;
				hit += origin;

				// ray triangle intersection
				// Is the intersection of the ray with the triangle's plane INSIDE the triangle?

				float4 ee1 = FetchFloat4(mesh->d_Triangles, 5 * idx + 2);
				//float4 ee1 = make_float4(cudaTriangleIntersectionData[20 * idx + 8], cudaTriangleIntersectionData[20 * idx + 9], cudaTriangleIntersectionData[20 * idx + 10], cudaTriangleIntersectionData[20 * idx + 11]);
				float kt1 = dot(ee1, hit) - ee1.w;
				if (kt1 < 0.0f) continue;

				float4 ee2 = FetchFloat4(mesh->d_Triangles, 5 * idx + 3);
				//float4 ee2 = make_float4(cudaTriangleIntersectionData[20 * idx + 12], cudaTriangleIntersectionData[20 * idx + 13], cudaTriangleIntersectionData[20 * idx + 14], cudaTriangleIntersectionData[20 * idx + 15]);
				float kt2 = dot(ee2, hit) - ee2.w;
				if (kt2 < 0.0f) continue;

				float4 ee3 = FetchFloat4(mesh->d_Triangles, 5 * idx + 4);
				//float4 ee3 = make_float4(cudaTriangleIntersectionData[20 * idx + 16], cudaTriangleIntersectionData[20 * idx + 17], cudaTriangleIntersectionData[20 * idx + 18], cudaTriangleIntersectionData[20 * idx + 19]);
				float kt3 = dot(ee3, hit) - ee3.w;
				if (kt3 < 0.0f) continue;

				// ray intersects triangle, "hit" is the world space coordinate of the intersection.
				{
					// is this intersection closer than all the others?
					float hitZ = distancesq(origin, hit);
					if (hitZ < bestTriDist) {

						// maintain the closest hit
						bestTriDist = hitZ;
						hitdist = sqrtf(bestTriDist);
						pBestTriIdx = idx;
						pointHitInWorldSpace = hit;

						// store barycentric coordinates (for texturing, not used for now)
						kAB = kt1;
						kBC = kt2;
						kCA = kt3;
					}
				}
			}
		}
	}

	return pBestTriIdx != -1;
}

class TriangleModel :public Hitable
{
public:
    __device__ TriangleModel(DevMaterial* mat, Suika::CudaTriangleModel* model) :Hitable(mat), mesh(model) {}
    __device__ ~TriangleModel() {}
    __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;

private:
	Suika::CudaTriangleModel* mesh;
};

__device__ bool TriangleModel::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
{
	Vector3Df pointHitInWorldSpace;
	float kAB = 0.f, kBC = 0.f, kCA = 0.f; // distances from the 3 edges of the triangle (from where we hit it), to be used for texturing

	int pBestTriIdx = -1;
	float tmin = 1e20;
	float tmax = -1e20;
	float d = 1e20;
	float scene_t = 1e20;
	float inf = 1e20;
	float hitdistance = 1e20;
	Vector3Df boxnormal = Vector3Df(0, 0, 0);

	// intersect all triangles in the scene stored in BVH
	BVH_IntersectTriangles(
		NULL,
		Vector3Df(r.pos.x, r.pos.y, r.pos.z),
		Vector3Df(r.dir.x, r.dir.y, r.dir.z),
		-1,
		pBestTriIdx, pointHitInWorldSpace, kAB, kBC, kCA, hitdistance, NULL,
		NULL, NULL, boxnormal, mesh);

	rec.p += rec.normal * 0.001;
	if (hitdistance < t_max && hitdistance > t_min)
	{
		rec.t = hitdistance;
		rec.p = make_float3(pointHitInWorldSpace.x, pointHitInWorldSpace.y, pointHitInWorldSpace.z);
		float4 normal = FetchFloat4(mesh->d_Triangles, 5 * pBestTriIdx + 1);
		rec.normal = make_float3(normal.x, normal.y, normal.z);
		rec.mat_ptr = material;
		return true;
	}
	else
	{
		return false;
	}
}
#endif