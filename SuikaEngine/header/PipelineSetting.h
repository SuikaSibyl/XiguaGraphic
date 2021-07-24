#pragma once
#include <Utility.h>
#include <Light.h>

static const int frameResourcesCount = 3;

struct ObjectConstants {
    DirectX::XMFLOAT4X4 world = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 texTransform = MathHelper::Identity4x4();

    //不同物体去索引对应材质
    UINT materialIndex = 0;//材质的索引
    UINT objPad0;	//占位
    UINT objPad1;	//占位
    UINT objPad2;	//占位
};

struct PassConstants
{
    // 0
    DirectX::XMFLOAT4X4 viewProj = MathHelper::Identity4x4();
    // 16
    XMFLOAT3 eyePos;
    float gTime = 0.0f;
    // 20
    XMFLOAT4 ambientLight;
    // 24
    DirectX::XMFLOAT4X4 gShadowTransform;
    // 40
    LightBasic light[16];
    // 56
    DirectX::XMFLOAT4X4 view = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 projection = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 viewInverse = MathHelper::Identity4x4();
};

struct MaterialData
{
    // 0
    XMFLOAT4 diffuseAlbedo = { 1.0f, 1.0f, 1.0f, 1.0f };//材质反照率
    // 4
    XMFLOAT3 fresnelR0 = { 0.01f, 0.01f, 0.01f };//RF(0)值，即材质的反射属性
    float roughness = 0.25f;//材质的粗糙度
    // 8
    XMFLOAT4X4 matTransform = MathHelper::Identity4x4();//纹理动画位移矩阵
    // 24
    UINT diffuseMapIndex = 0;//纹理数组索引
    UINT normalMapIndex;
    UINT extraMapIndex;
    UINT materialType;
    // 28
    //占位，向量数据打包时必须占满4位
    float metalness = 0.25f;//材质的粗糙度
    XMFLOAT3 emission = { 0.0f, 0.1f, 0.0f };//RF(0)值，即材质的反射属性
};