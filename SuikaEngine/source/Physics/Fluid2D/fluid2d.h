/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#ifndef __STABLEFLUIDS_KERNELS_CUH_
#define __STABLEFLUIDS_KERNELS_CUH_

#define DIM    16       // Square size of solver domain
#define DS    (DIM*DIM)  // Total domain size
#define CPADW (DIM/2+1)  // Padded width for real->complex in-place FFT
#define RPADW (2*(DIM/2+1))  // Padded width for real->complex in-place FFT
#define PDS   (DIM*CPADW) // Padded total domain size

#define DT     0.09f     // Delta T for interative solver
#define VIS    0.0025f   // Viscosity constant
#define FORCE (5.8f*DIM) // Force scale factor 
#define FR     4         // Force update radius

#define TILEX 64 // Tile width
#define TILEY 64 // Tile height
#define TIDSX 64 // Tids in X
#define TIDSY 4  // Tids in Y

 // Vector data type used to velocity and force fields
typedef float2 cData;
#include<DirectXMath.h>
using namespace DirectX;
#define M_PI 3.14159265358979323846

class Fluid2dSolver
{
public:
	Fluid2dSolver();
	~Fluid2dSolver();
	void OnUpdate(float time);
	void OnMouseInput(int x, int y);

	int TriangleNum()
	{
		return int(DS * 1.0 / 3);
	}

	int VertexNum()
	{
		return int(DS);
	}
	void SetTriangle(float* vertices, int i);
	float3 GetPosition(int i);
	float4 GetQuaternion(float3& pos);
	XMMATRIX GetTransform(int i);


private:
	cData* hvfield = NULL;
	cData* dvfield = NULL;

	cData* particles = NULL;
	cData* dparticles = NULL;

	cData* vxfield = NULL;
	cData* vyfield = NULL;

	void initParticles(cData* p, int dx, int dy);
	int lastx = 0, lasty = 0;
};

// This method adds constant force vectors to the velocity field
// stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
//__global__ void
//addForces_k(cData* v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch);

//// This method performs the velocity advection step, where we
//// trace velocity vectors back in time to update each grid cell.
//// That is, v(x,t+1) = v(p(x,-dt),t). Here we perform bilinear
//// interpolation in the velocity space.
//__global__ void
//advectVelocity_k(cData* v, float* vx, float* vy,
//    int dx, int pdx, int dy, float dt, int lb, cudaTextureObject_t tex);
//
//// This method performs velocity diffusion and forces mass conservation
//// in the frequency domain. The inputs 'vx' and 'vy' are complex-valued
//// arrays holding the Fourier coefficients of the velocity field in
//// X and Y. Diffusion in this space takes a simple form described as:
//// v(k,t) = v(k,t) / (1 + visc * dt * k^2), where visc is the viscosity,
//// and k is the wavenumber. The projection step forces the Fourier
//// velocity vectors to be orthogonal to the wave wave vectors for each
//// wavenumber: v(k,t) = v(k,t) - ((k dot v(k,t) * k) / k^2.
//__global__ void
//diffuseProject_k(cData* vx, cData* vy, int dx, int dy, float dt,
//    float visc, int lb);
//
//// This method updates the velocity field 'v' using the two complex
//// arrays from the previous step: 'vx' and 'vy'. Here we scale the
//// real components by 1/(dx*dy) to account for an unnormalized FFT.
//__global__ void
//updateVelocity_k(cData* v, float* vx, float* vy,
//    int dx, int pdx, int dy, int lb, size_t pitch);
//
//// This method updates the particles by moving particle positions
//// according to the velocity field and time step. That is, for each
//// particle: p(t+1) = p(t) + dt * v(p(t)).
//__global__ void
//advectParticles_k(cData* part, cData* v, int dx, int dy,
//    float dt, int lb, size_t pitch);

#endif