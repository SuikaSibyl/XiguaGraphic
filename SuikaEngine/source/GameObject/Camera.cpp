#include <Precompiled.h>
#include <Camera.h>
#include <QDirect3D12Widget.h>

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

void InteractSphere(float& u, float& v, XMVECTOR& pos, XMVECTOR& dir)
{
	XMFLOAT3 position, direction;
	XMStoreFloat3(&position, pos);
	XMStoreFloat3(&direction, dir);

	float a = direction.x * direction.x + 
		direction.y + direction.y + 
		direction.z * direction.z;

	float b = position.x * direction.x +
		position.y + direction.y +
		position.z * direction.z;
	b *= 2;

	float c = position.x * position.x +
		position.y + position.y +
		position.z * position.z - 9;

	float delta = b * b - 4 * a * c;
	if (delta > 0)
	{
		float root = sqrtf(delta);
		float distance = (b > 0) ? (-b + root) : (-b - root);
		distance /= 2 * a;
		XMVECTOR hitpoint = pos + distance * dir;

	}
	else
	{
		u = -1;
		v = -1;
	}
}

void Camera::OnMousePressed(QMouseEvent* event)
{
	int x = event->pos().x();
	int y = event->pos().y();
	XMVECTOR dir = getRayDir(x, y);
	float u, v;
	InteractSphere(u, v, pos, dir);
	mLastMousePosx = x;
	mLastMousePosy = y;
	
	Debug::Log(QString("x:") + QString::number(x) + QString(", y:") + QString::number(y));
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

XMVECTOR Camera::getRayDir(float s, float t)
{
	return XMVector3Normalize(lower_left_corner + s * 1. / m_pD3dWidget->width() * horizontal + t * 1. / m_pD3dWidget->width() * vertical - pos);
}

XMMATRIX Camera::GetViewMatrix()
{
	XMMATRIX view;

	//Convert Spherical to Cartesian coordinates. 
	float x = mRadius * cosf(-pitch) * cosf(-yaw);
	float z = mRadius * cosf(-pitch) * sinf(-yaw);
	float y = mRadius * sinf(-pitch);
	// Build the view matrix. 

	pos = XMVectorSet(pos_x, pos_y, pos_z, 1.0f);
	XMVECTOR target = XMVectorSet(pos_x + x, pos_y + y, pos_z + z, 1.0f);
	XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);


	XMVECTOR w = XMVector3Normalize(target - pos);
	XMVECTOR u = XMVector3Normalize(XMVector3Cross(up, w));
	XMVECTOR v = XMVector3Cross(w, u);

	lower_left_corner = pos - half_width * u - half_height * v + w;
	horizontal = 2 * half_width * u;
	vertical = 2 * half_height * v;

	view = XMMatrixLookAtLH(pos, target, up);

	if (camParas[0] != pos_x)
	{
		camParas[0] = pos_x;
		camUpdate = true;
	}
	if (camParas[1] != pos_y)
	{
		camParas[1] = pos_y;
		camUpdate = true;
	}
	if (camParas[2] != pos_z)
	{
		camParas[2] = pos_z;
		camUpdate = true;
	}
	if (camParas[3] != pos_x + x)
	{
		camParas[3] = pos_x + x;
		camUpdate = true;
	}
	if (camParas[4] != pos_y + y)
	{
		camParas[4] = pos_y + y;
		camUpdate = true;
	}
	if (camParas[5] != pos_z + z)
	{
		camParas[5] = pos_z + z;
		camUpdate = true;
	}

	return std::move(view);
}

void Camera::Init()
{
	camParas = new float[8];
	m_pInputSystem->AddListeningMem(InputSystem::InputSystem::Pause, this, &Camera::ToggleMode);
	m_pInputSystem->AddListeningMem(InputSystem::InputSystem::RTRender, this, &Camera::ToggleUseRT);

	float aspect = 1. * m_pD3dWidget->width() / m_pD3dWidget->height();
	float theta = vfov * M_PI / 180;
	half_height = tan(theta / 2);
	half_width = aspect * half_height;
	half_height *= -1;
	InitCubemapParas();
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

void Camera::ToggleUseRT()
{
	useRT = !useRT;
}