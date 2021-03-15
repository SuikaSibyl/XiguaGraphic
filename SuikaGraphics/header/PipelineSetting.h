#pragma once
#include <Utility.h>
#include <Light.h>

static const int frameResourcesCount = 3;

struct ObjectConstants {
    DirectX::XMFLOAT4X4 world = MathHelper::Identity4x4();
};

struct PassConstants
{
    DirectX::XMFLOAT4X4 viewProj = MathHelper::Identity4x4();
    XMFLOAT3 eyePos;
    float gTime = 0.0f;
    XMFLOAT4 ambientLight;
    Light light[16];
};