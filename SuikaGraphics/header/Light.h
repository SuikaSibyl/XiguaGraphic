#pragma once

#include <Utility.h>

struct Light {
	DirectX::XMFLOAT3 Strength; // Light color
	float FalloffStart; // point/spot light only
	DirectX::XMFLOAT3 Direction;// directional/spot light only
	float FalloffEnd; // point/spot light only
	DirectX::XMFLOAT3 Position; // point/spot light only
	float SpotPower; // spot light only
};

