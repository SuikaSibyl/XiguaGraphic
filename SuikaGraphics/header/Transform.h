#pragma once

#include <Utility.h>

using Microsoft::WRL::ComPtr;
using namespace DirectX;

class Transform
{
public:
	XMVECTOR position;
	XMVECTOR rotation;

	Transform()
	{
		position = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
		rotation = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
	}
};