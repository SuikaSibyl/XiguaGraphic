#pragma once

#include <Utility.h>
#include <InputSystem.h>
#include <Transform.h>

using Microsoft::WRL::ComPtr;
using namespace DirectX;

class QDirect3D12Widget;

class CudaManager;

class Camera
{
public:
	friend class CudaManager;
	float speed = 100;

	enum Mode {
		Focus,
		Defocus
	};

	XMFLOAT3 GetPosition()
	{
		XMFLOAT3 pos;
		pos.x = pos_x;
		pos.y = pos_y;
		pos.z = pos_z;
		return pos;
	}

	Mode mode = Focus;

	XMVECTOR position;
	XMVECTOR rotation;

	InputSystem* m_pInputSystem;
	QDirect3D12Widget* m_pD3dWidget;

	void SetMouseModeDefocus();
	void SetMouseModeFocus();

	void OnMousePressed(QMouseEvent* event);
	void OnMouseMove(QMouseEvent* event);

	void ToggleMode();
	void ToggleUseRT();

	bool DoUseRT()
	{
		return useRT;
	}

	XMMATRIX GetViewMatrix();

	void Update();

	void Init()
	{
		camParas = new float[8];
		m_pInputSystem->AddListeningMem(InputSystem::InputSystem::Pause, this, &Camera::ToggleMode);
		m_pInputSystem->AddListeningMem(InputSystem::InputSystem::RTRender, this, &Camera::ToggleUseRT);
	}

	DirectX::XMFLOAT4X4 mProj = MathHelper::Identity4x4();

private:
	int mLastMousePosx = 0;
	int mLastMousePosy = 0;

	float pitch = 0;
	float yaw = -DirectX::XM_PI / 2;

	float mTheta = 1.5f * DirectX::XM_PI;
	float mPhi = DirectX::XM_PIDIV4;
	float mRadius = 5.0f;

	float pos_x = 1;
	float pos_y = 0;
	float pos_z = 0;

	Transform transform;

	XMVECTOR GetInputTranslationDirection();

	// Ray Tracing Part
	bool useRT = false;
	bool camUpdate = true;
	float* camParas = nullptr;
};