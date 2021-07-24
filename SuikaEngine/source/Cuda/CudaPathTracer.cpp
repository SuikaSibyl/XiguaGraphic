#include <Precompiled.h>
#include <CudaPathTracer.h>

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
