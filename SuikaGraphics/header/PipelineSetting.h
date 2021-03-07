#pragma once
#include <Utility.h>

struct ObjectConstants {
    DirectX::XMFLOAT4X4 WorldViewProj = MathHelper::Identity4x4();
    float gTime = 0.0f;
    DirectX::XMFLOAT4X4 world = MathHelper::Identity4x4();
};

struct PassConstants
{
    DirectX::XMFLOAT4X4 viewProj = MathHelper::Identity4x4();
};