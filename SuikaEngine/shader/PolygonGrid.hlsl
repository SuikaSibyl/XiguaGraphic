
#ifndef __POLYGONGRID__
#define __POLYGONGRID__

#define WIDTH   (32)
#define HEIGHT  (32)
#define DEPTH   (32)
#define MAXINDEX (WIDTH*HEIGHT*DEPTH - 1)

#define BOUNDBOX 3

struct MyStruct
{
	float r,g,b;
    float4x4 SHCoeff_r;
    float4x4 SHCoeff_g;
    float4x4 SHCoeff_b;
};

uint Index3Dto1D(int3 index)
{
    uint i = index.x * WIDTH * HEIGHT + index.y * WIDTH + index.z;
    if(i<0) i=0;
    if(i>MAXINDEX) i = MAXINDEX;
	return i;
}

///
/// INPUT: position, each axis is in float[0, 31]
///
MyStruct TrilinearInterpolation(float3 position, StructuredBuffer<MyStruct> gPolygonSHData)
{
    float3 neighbor = int3(position);
    // between float[0, 1]
    float3 delta = position - neighbor;
    float tx = delta.r;
    float ty = delta.g;
    float tz = delta.b;
    float4x4 accum_r = 0;
    float4x4 accum_g = 0;
    float4x4 accum_b = 0;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
            {
                MyStruct test = gPolygonSHData[Index3Dto1D(neighbor + float3(i,j,k))];
                float4x4 coeffs_r = test.SHCoeff_r;
                float4x4 coeffs_g = test.SHCoeff_g;
                float4x4 coeffs_b = test.SHCoeff_b;

                float c = (i*tx+(1-i)*(1-tx))*
                    (j*ty+(1-j)*(1-ty))*
                    (k*tz+(1-k)*(1-tz));

                accum_r += c * coeffs_r;
                accum_g += c * coeffs_g;
                accum_b += c * coeffs_b;
            }

    MyStruct result;
    result.r=0;
    result.g=0;
    result.b=0;
    result.SHCoeff_r=accum_r;
    result.SHCoeff_g=accum_g;
    result.SHCoeff_b=accum_b;
    return result;
}

float3 Grid2Position(float3 grid)
{
    grid /= float3(WIDTH,HEIGHT,DEPTH);
    grid *= float3(BOUNDBOX*2,BOUNDBOX*2,BOUNDBOX*2);
    grid -= float3(BOUNDBOX, BOUNDBOX, BOUNDBOX);
    return grid;
}

MyStruct PositionInterpo(float3 position, StructuredBuffer<MyStruct> gPolygonSHData)
{
    float3 pos = position + float3(BOUNDBOX, BOUNDBOX, BOUNDBOX);
    pos /= float3(BOUNDBOX*2,BOUNDBOX*2,BOUNDBOX*2);

    pos *= float3(WIDTH,HEIGHT,DEPTH);
    
    // MyStruct coeff = gPolygonSHData[Index3Dto1D(pos)];

    MyStruct coeff = TrilinearInterpolation(pos, gPolygonSHData);
    
    return coeff;
}

#endif