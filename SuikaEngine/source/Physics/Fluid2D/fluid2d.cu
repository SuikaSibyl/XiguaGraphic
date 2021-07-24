#include <cufft.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <CudaUtil.h>
#include <helper_math.h>
#include "fluid2d.h"
#pragma comment(lib,"cufft.lib")
//Texture object for reading velocity field
cudaTextureObject_t     texObj;
static cudaArray* array = NULL;

//// Particle data
//extern GLuint vbo;                 // OpenGL vertex buffer object
//extern struct cudaGraphicsResource* cuda_vbo_resource; // handles OpenGL-CUDA exchange

// Texture pitch
size_t tPitch;
cufftHandle planr2c;
cufftHandle planc2r;

// Note that these kernels are designed to work with arbitrary
// domain sizes, not just domains that are multiples of the tile
// size. Therefore, we have extra code that checks to make sure
// a given thread location falls within the domain boundaries in
// both X and Y. Also, the domain is covered by looping over
// multiple elements in the Y direction, while there is a one-to-one
// mapping between threads in X and the tile size in X.
// Nolan Goodnight 9/22/06

// This method adds constant force vectors to the velocity field
// stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
__global__ void addForces_k(cData* v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    cData* fj = (cData*)((char*)v + (ty + spy) * pitch) + tx + spx;

    cData vterm = *fj;
    tx -= r;
    ty -= r;
    float s = 1.f / (1.f + tx * tx * tx * tx + ty * ty * ty * ty);
    vterm.x += s * fx;
    vterm.y += s * fy;
    *fj = vterm;
}

// This method performs the velocity advection step, where we
// trace velocity vectors back in time to update each grid cell.
// That is, v(x,t+1) = v(p(x,-dt),t). Here we perform bilinear
// interpolation in the velocity space.
__global__ void
advectVelocity_k(cData* v, float* vx, float* vy,
    int dx, int pdx, int dy, float dt, int lb, cudaTextureObject_t texObject)
{
    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    cData vterm, ploc;
    float vxterm, vyterm;

    // gtidx is the domain location in x for this thread
    if (gtidx < dx)
    {
        for (p = 0; p < lb; p++)
        {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;

            if (fi < dy)
            {
                int fj = fi * pdx + gtidx;
                vterm = tex2D<cData>(texObject, (float)gtidx, (float)fi);
                ploc.x = (gtidx + 0.5f) - (dt * vterm.x * dx);
                ploc.y = (fi + 0.5f) - (dt * vterm.y * dy);
                vterm = tex2D<cData>(texObject, ploc.x, ploc.y);
                vxterm = vterm.x;
                vyterm = vterm.y;
                vx[fj] = vxterm;
                vy[fj] = vyterm;
            }
        }
    }
}

// This method performs velocity diffusion and forces mass conservation
// in the frequency domain. The inputs 'vx' and 'vy' are complex-valued
// arrays holding the Fourier coefficients of the velocity field in
// X and Y. Diffusion in this space takes a simple form described as:
// v(k,t) = v(k,t) / (1 + visc * dt * k^2), where visc is the viscosity,
// and k is the wavenumber. The projection step forces the Fourier
// velocity vectors to be orthogonal to the vectors for each
// wavenumber: v(k,t) = v(k,t) - ((k dot v(k,t) * k) / k^2.
__global__ void
diffuseProject_k(cData* vx, cData* vy, int dx, int dy, float dt,
    float visc, int lb)
{

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    cData xterm, yterm;

    // gtidx is the domain location in x for this thread
    if (gtidx < dx)
    {
        for (p = 0; p < lb; p++)
        {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;

            if (fi < dy)
            {
                int fj = fi * dx + gtidx;
                xterm = vx[fj];
                yterm = vy[fj];

                // Compute the index of the wavenumber based on the
                // data order produced by a standard NN FFT.
                int iix = gtidx;
                int iiy = (fi > dy / 2) ? (fi - (dy)) : fi;

                // Velocity diffusion
                float kk = (float)(iix * iix + iiy * iiy); // k^2
                float diff = 1.f / (1.f + visc * dt * kk);
                xterm.x *= diff;
                xterm.y *= diff;
                yterm.x *= diff;
                yterm.y *= diff;

                // Velocity projection
                if (kk > 0.f)
                {
                    float rkk = 1.f / kk;
                    // Real portion of velocity projection
                    float rkp = (iix * xterm.x + iiy * yterm.x);
                    // Imaginary portion of velocity projection
                    float ikp = (iix * xterm.y + iiy * yterm.y);
                    xterm.x -= rkk * rkp * iix;
                    xterm.y -= rkk * ikp * iix;
                    yterm.x -= rkk * rkp * iiy;
                    yterm.y -= rkk * ikp * iiy;
                }

                vx[fj] = xterm;
                vy[fj] = yterm;
            }
        }
    }
}

// This method updates the velocity field 'v' using the two complex
// arrays from the previous step: 'vx' and 'vy'. Here we scale the
// real components by 1/(dx*dy) to account for an unnormalized FFT.
__global__ void
updateVelocity_k(cData* v, float* vx, float* vy,
    int dx, int pdx, int dy, int lb, size_t pitch)
{

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    float vxterm, vyterm;
    cData nvterm;

    // gtidx is the domain location in x for this thread
    if (gtidx < dx)
    {
        for (p = 0; p < lb; p++)
        {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;

            if (fi < dy)
            {
                int fjr = fi * pdx + gtidx;
                vxterm = vx[fjr];
                vyterm = vy[fjr];

                // Normalize the result of the inverse FFT
                float scale = 1.f / (dx * dy);
                nvterm.x = vxterm * scale;
                nvterm.y = vyterm * scale;

                cData* fj = (cData*)((char*)v + fi * pitch) + gtidx;
                *fj = nvterm;
            }
        } // If this thread is inside the domain in Y
    } // If this thread is inside the domain in X
}
// This method updates the particles by moving particle positions
// according to the velocity field and time step. That is, for each
// particle: p(t+1) = p(t) + dt * v(p(t)).
__global__ void advectParticles_k(cData* part, cData* v, int dx, int dy,
    float dt, int lb, size_t pitch)
{

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    // gtidx is the domain location in x for this thread
    cData pterm, vterm;

    if (gtidx < dx)
    {
        for (p = 0; p < lb; p++)
        {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;

            if (fi < dy)
            {
                int fj = fi * dx + gtidx;
                pterm = part[fj];

                int xvi = ((int)(pterm.x * dx));
                int yvi = ((int)(pterm.y * dy));
                vterm = *((cData*)((char*)v + yvi * pitch) + xvi);

                pterm.x += dt * vterm.x;
                pterm.x = pterm.x - (int)pterm.x;
                pterm.x += 1.f;
                pterm.x = pterm.x - (int)pterm.x;
                pterm.y += dt * vterm.y;
                pterm.y = pterm.y - (int)pterm.y;
                pterm.y += 1.f;
                pterm.y = pterm.y - (int)pterm.y;

                part[fj] = pterm;
            }
        } // If this thread is inside the domain in Y
    } // If this thread is inside the domain in X
}


// These are the external function calls necessary for launching fluid simulation
void addForces(cData * v, int dx, int dy, int spx, int spy, float fx, float fy, int r)
{
    dim3 tids(2 * r + 1, 2 * r + 1);

    addForces_k <<<1, tids >>> (v, dx, dy, spx, spy, fx, fy, r, tPitch);
    checkCudaErrors(cudaGetLastError());
}

void updateTexture(cData* data, size_t wib, size_t h, size_t pitch)
{
    checkCudaErrors(cudaMemcpy2DToArray(array, 0, 0, data, pitch, wib, h, cudaMemcpyDeviceToDevice));
}

void advectVelocity(cData * v, float* vx, float* vy, int dx, int pdx, int dy, float dt)
{
    dim3 grid((dx / TILEX) + (!(dx % TILEX) ? 0 : 1), (dy / TILEY) + (!(dy % TILEY) ? 0 : 1));

    dim3 tids(TIDSX, TIDSY);

    updateTexture(v, DIM * sizeof(cData), DIM, tPitch);
    advectVelocity_k << <grid, tids >> > (v, vx, vy, dx, pdx, dy, dt, TILEY / TIDSY, texObj);

    checkCudaErrors(cudaGetLastError());
}


void diffuseProject(cData * vx, cData * vy, int dx, int dy, float dt, float visc)
{
    // Forward FFT
    cufftExecR2C(planr2c, (cufftReal*)vx, (cufftComplex*)vx);
    cufftExecR2C(planr2c, (cufftReal*)vy, (cufftComplex*)vy);

    uint3 grid = make_uint3((dx / TILEX) + (!(dx % TILEX) ? 0 : 1),
        (dy / TILEY) + (!(dy % TILEY) ? 0 : 1), 1);
    uint3 tids = make_uint3(TIDSX, TIDSY, 1);

    diffuseProject_k << <grid, tids >> > (vx, vy, dx, dy, dt, visc, TILEY / TIDSY);
    checkCudaErrors(cudaGetLastError());

    // Inverse FFT
    cufftExecC2R(planc2r, (cufftComplex*)vx, (cufftReal*)vx);
    cufftExecC2R(planc2r, (cufftComplex*)vy, (cufftReal*)vy);
}

void updateVelocity(cData * v, float* vx, float* vy, int dx, int pdx, int dy)
{
    dim3 grid((dx / TILEX) + (!(dx % TILEX) ? 0 : 1), (dy / TILEY) + (!(dy % TILEY) ? 0 : 1));
    dim3 tids(TIDSX, TIDSY);

    updateVelocity_k << <grid, tids >> > (v, vx, vy, dx, pdx, dy, TILEY / TIDSY, tPitch);
    checkCudaErrors(cudaGetLastError());
}

void advectParticles(cData* dparticles, cData * v, int dx, int dy, float dt)
{
    dim3 grid((dx / TILEX) + (!(dx % TILEX) ? 0 : 1), (dy / TILEY) + (!(dy % TILEY) ? 0 : 1));
    dim3 tids(TIDSX, TIDSY);

    advectParticles_k << <grid, tids >> > (dparticles, v, dx, dy, dt, TILEY / TIDSY, tPitch);
    checkCudaErrors(cudaGetLastError());
}

void setupTexture(int x, int y)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

    cudaMallocArray(&array, &desc, y, x);
    checkCudaErrors(cudaGetLastError());

    cudaResourceDesc            texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = array;

    cudaTextureDesc             texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL));
}


Fluid2dSolver::Fluid2dSolver()
{
    // host velocity filed malloc
    hvfield = (cData*)malloc(sizeof(cData) * DS);
    memset(hvfield, 0, sizeof(cData) * DS);
    // Allocate and initialize device data
    cudaMallocPitch((void**)&dvfield, &tPitch, sizeof(cData) * DIM, DIM);
    // Init device velocity field
    cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS, cudaMemcpyHostToDevice);

    // Temporary complex velocity field data
    cudaMalloc((void**)&vxfield, sizeof(cData) * PDS);
    cudaMalloc((void**)&vyfield, sizeof(cData) * PDS);

    // Set up Texture
    setupTexture(DIM, DIM);

    // Create particle array
    particles = (cData*)malloc(sizeof(cData) * DS);
    memset(particles, 0, sizeof(cData) * DS);
    // Allocate and initialize device data
    cudaMalloc((void**)&dparticles, sizeof(cData) * DS);
    // Set up particles
    initParticles(particles, DIM, DIM);
    // Init device velocity field
    cudaMemcpy(dparticles, particles, sizeof(cData) * DS, cudaMemcpyHostToDevice);

    // Create CUFFT transform plan configuration
    cufftPlan2d(&planr2c, DIM, DIM, CUFFT_R2C);
    cufftPlan2d(&planc2r, DIM, DIM, CUFFT_C2R);
}

void Fluid2dSolver::OnUpdate(float time)
{
    // simulate fluid
    addForces(dvfield, DIM, DIM, time*0.0000005, time * 0.0000005, FORCE * DT * 0.00001, FORCE * DT * 0.00001, FR);
    advectVelocity(dvfield, (float*)vxfield, (float*)vyfield, DIM, RPADW, DIM, DT);
    diffuseProject(vxfield, vyfield, CPADW, DIM, DT, VIS);
    updateVelocity(dvfield, (float*)vxfield, (float*)vyfield, DIM, RPADW, DIM);
    advectParticles(dparticles, dvfield, DIM, DIM, DT);
    cudaMemcpy(particles, dparticles, sizeof(cData) * DS, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void Fluid2dSolver::OnMouseInput(int x, int y)
{
    //// Convert motion coordinates to domain
    //float fx = (lastx / (float)wWidth);
    //float fy = (lasty / (float)wHeight);
    //int nx = (int)(fx * DIM);
    //int ny = (int)(fy * DIM);

    //if (clicked && nx < DIM - FR && nx > FR - 1 && ny < DIM - FR && ny > FR - 1)
    //{
    //    int ddx = x - lastx;
    //    int ddy = y - lasty;
    //    fx = ddx / (float)wWidth;
    //    fy = ddy / (float)wHeight;
    //    int spy = ny - FR;
    //    int spx = nx - FR;
    //    addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);
    //    lastx = x;
    //    lasty = y;
    //}
}

float3 Float2ToFloat3(float2 uv)
{
    float theta = uv.x * M_PI * 0.5;
    float phi = (uv.y - 0.5) * M_PI * 2;
    float3 res;
    res.x = cosf(theta) * cosf(phi);
    res.y = sinf(theta);
    res.z = cosf(theta) * sinf(phi);
    return res;
}

float MinDistance_seg(float3& point, float3& v1, float3& v2)
{
    float l = length((v2 - v1));
    float cast = dot((v2 - v1), (point - v1));
    cast /= l * l;
    if (cast < 0)
    {
        return length((point - v1));
    }
    else if (cast > 1)
    {
        return length((point - v2));
    }
    return length(point - v1 - cast * ((v2 - v1)));
}

float MinDistance(float3& point, float3& v1, float3& v2, float3& v3)
{
    float3 normal = normalize(cross((v2 - v1), (v3 - v1)));
    float distance = fabs(dot(normal, (point - v1)));
    return distance;
}

void Fluid2dSolver::SetTriangle(float* vertices, int i)
{
    float2 point1 = particles[i * 3 + 0];
    float2 point2 = particles[i * 3 + 1];
    float2 point3 = particles[i * 3 + 2];

    float3 pos1 = Float2ToFloat3(point1);
    float3 pos2 = Float2ToFloat3(point2);
    float3 pos3 = Float2ToFloat3(point3);

    float3 origin = make_float3(0, 0, 0);
    float l = 10. / MinDistance(origin, pos1, pos2, pos3);
    pos1 *= l; pos2 *= l; pos3 *= l;
    vertices[0] = pos1.x; vertices[1] = pos1.y; vertices[2] = pos1.z;
    vertices[3] = pos2.x; vertices[4] = pos2.y; vertices[5] = pos2.z;
    vertices[6] = pos3.x; vertices[6] = pos3.y; vertices[8] = pos3.z;
}

float3 Fluid2dSolver::GetPosition(int i)
{
    float2 point = particles[i];
    return Float2ToFloat3(point);
}