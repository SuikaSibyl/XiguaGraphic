#include <Camera.h>
#include <QDirect3D12Widget.h>
#include <MathHelper.h>
#include <Utility.h>

void Camera::SetMouseModeDefocus()
{
	m_pD3dWidget->setMouseTracking(false);
	m_pD3dWidget->setCursor(Qt::ArrowCursor);
}

void Camera::SetMouseModeFocus()
{
	m_pD3dWidget->setMouseTracking(true);
	m_pD3dWidget->setCursor(Qt::BlankCursor);
}

void Camera::OnMousePressed(QMouseEvent* event)
{
	int x = event->pos().x();
	int y = event->pos().y();

	mLastMousePosx = x;
	mLastMousePosy = y;
}

void Camera::OnMouseMove(QMouseEvent* event)
{
	int x = event->pos().x();
	int y = event->pos().y();

	float dx = XMConvertToRadians(0.25f * static_cast<float> (x - mLastMousePosx));
	float dy = XMConvertToRadians(0.25f * static_cast<float> (y - mLastMousePosy));

	yaw += dx;
	pitch += dy;

	if (pitch > DirectX::XM_PI / 2)
		pitch = DirectX::XM_PI / 2 -0.00001f;
	if (pitch < -DirectX::XM_PI / 2)
		pitch = -DirectX::XM_PI / 2 + 0.00001f;

	switch (mode)
	{
	case Camera::Focus:
	{
		QPoint glob = m_pD3dWidget->mapToGlobal(QPoint(m_pD3dWidget->width() / 2, m_pD3dWidget->height() / 2));
		QCursor::setPos(glob);
		mLastMousePosx = m_pD3dWidget->width() / 2;
		mLastMousePosy = m_pD3dWidget->height() / 2;
		break;
	}
	case Camera::Defocus:
	{
		mLastMousePosx = x;
		mLastMousePosy = y;
		break;
	}
	default:
		break;
	}
}

XMMATRIX& Camera::GetViewMatrix()
{
	XMMATRIX view;

	//Convert Spherical to Cartesian coordinates. 
	float x = mRadius * cosf(-pitch) * cosf(-yaw);
	float z = mRadius * cosf(-pitch) * sinf(-yaw);
	float y = mRadius * sinf(-pitch);
	// Build the view matrix. 

	XMVECTOR pos = XMVectorSet(pos_x, pos_y, pos_z, 1.0f);
	XMVECTOR target = XMVectorSet(pos_x + x, pos_y + y, pos_z + z, 1.0f);
	XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
	view = XMMatrixLookAtLH(pos, target, up);

	return view;
}

void Camera::Update()
{
	XMVECTOR translation = GetInputTranslationDirection() * GameTimer::DeltaTime() * speed;
	DirectX::XMMATRIX M = XMMatrixRotationRollPitchYaw(0, yaw, -pitch);
	translation = XMVector4Transform(translation, M);
	XMFLOAT3 move;
	XMStoreFloat3(&move, translation);
	pos_x += move.x;
	pos_y += move.y;
	pos_z += move.z;
}

XMVECTOR Camera::GetInputTranslationDirection()
{
	XMVECTOR direction = XMVectorZero();

	if (m_pInputSystem->KeyboardPressed[InputSystem::InputTypes::Forward])
	{
		direction += XMVectorSet(1, 0, 0, 0);
	}
	if (m_pInputSystem->KeyboardPressed[InputSystem::InputTypes::Back])
	{
		direction += XMVectorSet(-1, 0, 0, 0);
	}
	if (m_pInputSystem->KeyboardPressed[InputSystem::InputTypes::Left])
	{
		direction += XMVectorSet(0, 0, 1, 0);
	}
	if (m_pInputSystem->KeyboardPressed[InputSystem::InputTypes::Right])
	{
		direction += XMVectorSet(0, 0, -1, 0);
	}
	if (m_pInputSystem->KeyboardPressed[InputSystem::InputTypes::Up])
	{
		direction += XMVectorSet(0, 1, 0, 0);
	}
	if (m_pInputSystem->KeyboardPressed[InputSystem::InputTypes::Down])
	{
		direction += XMVectorSet(0, -1, 0, 0);
	}

	return direction;
}

void Camera::ToggleMode()
{
	if (mode == Focus)
	{
		mode = Defocus;
		SetMouseModeDefocus();
	}
	else if (mode == Defocus)
	{
		mode = Focus;
		SetMouseModeFocus();
	}
}