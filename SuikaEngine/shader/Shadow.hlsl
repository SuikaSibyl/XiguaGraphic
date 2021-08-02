#include "ShaderLibrary/Common.hlsl"

struct VertexOut
{
	float4 PosH    : SV_POSITION;
	float2 TexC    : TEXCOORD;
};

VertexOut VS(VertexIn vin)
{
	VertexOut vout = (VertexOut)0.0f;
	
    // Transform to world space.
    float4 posW = mul(float4(vin.PosL, 1.0f), gWorld);

    // Transform to homogeneous clip space.
    vout.PosH = mul(posW, gViewProj);
	
	// Output vertex attributes for interpolation across triangle.
	float4 texC = mul(float4(vin.TexC, 0.0f, 1.0f), gTexTrans);
    vout.TexC = texC.xy;

    return vout;
}

// This is only used for alpha cut out geometry, so that shadows 
// show up correctly.  Geometry that does not need to sample a
// texture can use a NULL pixel shader for depth pass.
void PS(VertexOut pin) 
{
    if(materialIndex==-1)
        return;
        
	// Fetch the material data.
	MaterialData matData = gMaterialData[materialIndex];
	float4 diffuseAlbedo = matData.gDiffuseAlbedo;
    uint diffuseMapIndex = matData.gDiffuseMapIndex;
	
	// Dynamically look up the texture in the array.
	diffuseAlbedo *= gDiffuseMap[diffuseMapIndex].Sample(gSamAnisotropicWarp, pin.TexC);

#ifdef ALPHA_TEST
    // Discard pixel if texture alpha < 0.1.  We do this test as soon 
    // as possible in the shader so that we can potentially exit the
    // shader early, thereby skipping the rest of the shader code.
    clip(diffuseAlbedo.a - 0.1f);
#endif
}