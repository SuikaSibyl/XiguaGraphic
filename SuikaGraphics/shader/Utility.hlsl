static float PI = 3.14159265359;

float3 LerpFloat3(float3 left, float3 right, float alpha)
{
    return (1-alpha)*left+alpha*right;
}

float3 ReinhardHDR(float3 color)
{
    color = color / (color + float3(1.0,1.0,1.0));
    color = pow(color, float3(1.0/2.2,1.0/2.2,1.0/2.2)); 
    return color;
}

float atan2f(float y, float x)
{
    float base = atan(y/x);
    if(x<0)
    {
        if(y>0)
        {
            base = base + PI;
        }
        else
        {
            base = base - PI;
        }
    }
    return base;
}

static float2 invAtan = float2(0.1591, 0.3183);
float2 SampleSphericalMap(float3 v)
{
    float2 uv = float2(atan2f(v.z,v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}