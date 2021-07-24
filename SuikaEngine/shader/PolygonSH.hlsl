
#include "PRT.hlsl"
#include "PolygonGrid.hlsl"
#include "PRT.hlsl"

#define N 256  //一个线程组有256个线程

cbuffer cbSettings : register(b0)
{
    int lightNum;

	// Support up to 11 blur weights.
    //11个权重（由于是根常量传过来的，所以不能用数组）
    float w0;
    float w1;
    float w2;
    float w3;
    float w4;
    float w5;
    float w6;
    float w7;
    float w8;
    float w9;
    float w10;    
};

// Texture2D gInput : register(t0);    //输入的SRV纹理
struct Vertex
{
	float3 PosL  : POSITION;
    float3 Normal : NORMAL;
    float2 TexC : TEXCOORD;
    row_major float4x4 Transfer : TRANSFER;
    row_major float4x4 Thiness : THINESS;
};

StructuredBuffer<Vertex> gMaterialData : register(t0, space1);
RWStructuredBuffer<MyStruct> myStructuredBuffer : register(u0);


[numthreads(8, 8, 8)] //线程数定义（纵向）
void CS(int3 groupThreadID : SV_GroupThreadID, //组内线程ID
        int3 dispatchThreadID : SV_DispatchThreadID)//分派ID
{
	uint index = Index3Dto1D(dispatchThreadID);
	if(index>=WIDTH*HEIGHT*DEPTH) return;

	float4x4 LightCoeff_r = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	float4x4 LightCoeff_g = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	float4x4 LightCoeff_b = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	float3 PosW = Grid2Position(dispatchThreadID);

	for(int i=0; i<lightNum; i++)
	{
		Vertex v1 = gMaterialData[i*6 + 3];
		Vertex v2 = gMaterialData[i*6 + 4];
		Vertex v3 = gMaterialData[i*6 + 5];
		float3 ver[3] = {v1.PosL, v2.PosL, v3.PosL};
		float4x4 tmp = ComputeCoefficients(PosW, ver, 3);
		LightCoeff_r += v1.Transfer[0][0] * tmp;
		LightCoeff_g += v1.Transfer[0][1] * tmp;
		LightCoeff_b += v1.Transfer[0][2] * tmp;
	}

	MyStruct test;
	test.r=1. * dispatchThreadID.x / WIDTH;
	test.g=1. * dispatchThreadID.y / HEIGHT;
	test.b=1. * dispatchThreadID.z / DEPTH;
	test.SHCoeff_r = LightCoeff_r;
	test.SHCoeff_g = LightCoeff_g;
	test.SHCoeff_b = LightCoeff_b;
	
    myStructuredBuffer[index] = test;
}