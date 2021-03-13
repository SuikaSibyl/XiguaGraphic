#pragma once

#include <Utility.h>
#include <InputSystem.h>

using Microsoft::WRL::ComPtr;
using namespace DirectX;

class Camera
{
public:
	XMVECTOR position;
	XMVECTOR rotation;

	InputSystem* m_pInputSystem;
};