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