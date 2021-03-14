#pragma once

#include <Utility.h>

using Microsoft::WRL::ComPtr;
using namespace DirectX;

class Transform
{
public:
	Transform()
	{
		local_position = XMFLOAT3(0.0f, 0.0f, 0.0f);
		local_rotation = XMFLOAT3(0.0f, 0.0f, 0.0f);
		parent = nullptr;
	}

	XMVECTOR GetPosition()const
	{
		return XMLoadFloat3(&local_position);
	}

	XMFLOAT3 GetPosition3f()const
	{
		return local_position;
	}

	Transform* parent;
private:
	DirectX::XMFLOAT3 local_position;
	DirectX::XMFLOAT3 local_rotation;
};