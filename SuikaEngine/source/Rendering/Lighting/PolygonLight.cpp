#include <Precompiled.h>
#include "PolygonLight.h"

PolygonLight::PolygonLight(float* points, float* color, int vnum)
{
	for (int i = 0; i < vnum; i++)
	{
		m_Points.emplace_back(points[0 + i * 3], points[1 + i * 3], points[2 + i * 3]);
	}
	m_Color = Color(color[0], color[1], color[2]);
}

void PolygonLight::ResetPoint(float* points)
{
	m_Points.clear();
	for (int i = 0; i < 3; i++)
	{
		m_Points.emplace_back(points[0 + i * 3], points[1 + i * 3], points[2 + i * 3]);
	}
}

XMFLOAT3 PolygonLight::GetPosition(int n)
{
	return XMFLOAT3(m_Points[n].x, m_Points[n].y, m_Points[n].z);
}

XMFLOAT3 PolygonLight::GetColor(int n)
{
	return XMFLOAT3(m_Color.r, m_Color.g, m_Color.b);
}

XMMATRIX& PolygonLight::GetTransform()
{
	return transform;
}

std::vector<uint16>& PolygonLightStack::GetIndices16()
{
	uint lightNum = CountLights();
	if (mIndices16.empty())
	{
		for (uint i = 0; i < lightNum; i++)
		{
			mIndices16.push_back(i * 6 + 0);
			mIndices16.push_back(i * 6 + 1);
			mIndices16.push_back(i * 6 + 2);
			mIndices16.push_back(i * 6 + 3);
			mIndices16.push_back(i * 6 + 5);
			mIndices16.push_back(i * 6 + 4);
		}
	}
	return mIndices16;
}

std::vector<Vertex>& PolygonLightStack::GetVertex()
{
	uint lightNum = CountLights();
	uint index = 0;
	for (uint i = 0; i < lightNum; i++)
	{
		for (int n = 0; n < 3; n++)
		{
			Vertex v;
			v.Pos = GetPosition(i, n);
			v.Normal = XMFLOAT3(0, 0, 0);
			mVertex.push_back(std::move(v));
		}
		for (int n = 0; n < 3; n++)
		{
			Vertex v;
			v.Pos = GetPosition(i, n);
			v.Normal = GetColor(i, n);
			mVertex.push_back(std::move(v));
		}
	}
	return mVertex;
}

void PolygonLightStack::PushPolygon(float* points, float* color, int vnum)
{
	m_Lights.emplace_back(points, color, vnum);
}

uint PolygonLightStack::CountLights()
{
	return m_Lights.size();
}
uint PolygonLightStack::CountVertex()
{
	return m_Lights.size() * 6;
}

XMFLOAT3& PolygonLightStack::GetPosition(uint i, int n)
{
	return m_Lights[i].GetPosition(n);
}

XMFLOAT3& PolygonLightStack::GetColor(uint i, int n)
{
	return m_Lights[i].GetColor(n);
}

XMMATRIX& PolygonLightStack::GetTransform(uint i)
{
	return m_Lights[i].GetTransform();
}

float PolygonLightStack::SetTransform(uint i, XMMATRIX& matrix)
{
	XMMATRIX& transform = GetTransform(i);
	transform = std::move(matrix);
	return transform.r[3].m128_f32[0];
}

void PolygonLightStack::UpdateDynamicVB(UploadBuffer<Geometry::Vertex>* dynamicVB)
{
	uint lightNum = CountLights();
	uint index = 0;
	for (uint i = 0; i < lightNum; i++)
	{
		DirectX::XMVECTOR vertex[3] = { 
			DirectX::XMLoadFloat3(&(GetPosition(i, 0))),
			DirectX::XMLoadFloat3(&(GetPosition(i, 1))),
			DirectX::XMLoadFloat3(&(GetPosition(i, 2))) };

		if (UseTransform)
		{
			XMMATRIX& transform = GetTransform(i);
			vertex[0] = XMVector3TransformCoord(vertex[0], transform);
			vertex[1] = XMVector3TransformCoord(vertex[1], transform);
			vertex[2] = XMVector3TransformCoord(vertex[2], transform);
		}

		DirectX::XMVECTOR normal_vec = XMVector3Cross(vertex[1] - vertex[0], vertex[2] - vertex[0]);
		DirectX::XMVECTOR normal_vec_inv = normal_vec * -1;
		XMFLOAT3 normal, inv_normal;
		XMStoreFloat3(&normal, normal_vec);
		XMStoreFloat3(&inv_normal, normal_vec_inv);
		for (int n = 0; n < 3; n++)
		{
			Vertex v;
			XMStoreFloat3(&v.Pos, vertex[n]);
			//v.Pos = GetPosition(i, n);
			v.Normal = normal;
			v.SHTransfer = XMMATRIX(0, 0, 0, 0, 
									0, 0, 0, 0, 
									0, 0, 0, 0, 
									0, 0, 0, 0);
			dynamicVB->CopyData(index++, v);
		}
		for (int n = 0; n < 3; n++)
		{
			XMFLOAT3 color = GetColor(i, n);
			Vertex v;
			XMStoreFloat3(&v.Pos, vertex[n]);
			//v.Pos = GetPosition(i, n);
			v.Normal = inv_normal;
			v.SHTransfer = XMMATRIX(color.x, color.y, color.z, 0,
									0, 0, 0, 0,
									0, 0, 0, 0,
									0, 0, 0, 0);
			dynamicVB->CopyData(index++, v);
		}
	}
}