#include <Precompiled.h>
#include "fluid2d.h"

Fluid2dSolver::~Fluid2dSolver()
{
}

// very simple von neumann middle-square prng.  can't use rand() in -qatest
// mode because its implementation varies across platforms which makes testing
// for consistency in the important parts of this program difficult.
float myrand(void)
{
    return rand() / (float)RAND_MAX;
}

void Fluid2dSolver::initParticles(cData* p, int dx, int dy)
{
    int i, j;
    for (i = 0; i < dy; i++)
    {
        for (j = 0; j < dx; j++)
        {
            p[i * dx + j].x = (j + 0.5f + (myrand() - 0.5f)) / dx;
            p[i * dx + j].y = (i + 0.5f + (myrand() - 0.5f)) / dy;
        }
    }
}

XMFLOAT3 scale(1, 1, 1);
XMFLOAT3 origin(0, 0, 0);

XMMATRIX Fluid2dSolver::GetTransform(int i)
{
    float3 pos = GetPosition(i);
    XMVECTOR position = XMVectorSet(3 * pos.x, 3 * pos.y, 3 * pos.z, 0.0f);
    XMVECTOR direction = XMVectorSet(-pos.x, -pos.y, -pos.z, 0.0f);
    XMVECTOR normal = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

    //vector a = crossproduct(v1, v2);
    //q.xyz = a;
    //q.w = sqrt((v1.Length ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2);
    XMVECTOR n = XMVector3Cross(normal, direction);
    XMFLOAT3 xyz;
    XMStoreFloat3(&xyz, n);
    float w = 1 - pos.y;
    XMVECTOR rotation = XMVectorSet(xyz.x, xyz.y, xyz.z, w);
    rotation = XMVector4Normalize(rotation);

    return XMMatrixAffineTransformation(
        XMLoadFloat3(&scale),
        XMLoadFloat3(&origin),
        rotation,
        position);
}