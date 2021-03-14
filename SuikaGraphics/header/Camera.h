#pragma once

#include <Utility.h>
#include <InputSystem.h>
#include <Transform.h>

using Microsoft::WRL::ComPtr;
using namespace DirectX;

class QDirect3D12Widget;

class Camera
{
public:
	float speed = 100;

	enum Mode {
		Focus,
		Defocus
	};

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

	XMMATRIX& GetViewMatrix();

	void Update();

	void Init()
	{
		m_pInputSystem->AddListeningMem(InputSystem::InputSystem::Pause, this, &Camera::ToggleMode);
	}

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
};