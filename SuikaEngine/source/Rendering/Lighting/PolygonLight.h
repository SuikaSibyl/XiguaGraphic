#pragma once
#include <MeshGeometry.h>

using uint16 = std::uint16_t;
using Geometry::Vertex;

struct PolygonLight
{

	struct Point
	{
		float x;
		float y;
		float z;

		Point() = default;
		Point(float xx, float yy, float zz) :
			x(xx), y(yy), z(zz) {}
	};

	struct Color
	{
		float r;
		float g;
		float b;

		Color() = default;
		Color(float rr, float gg, float bb) :
			r(rr), g(gg), b(bb) {}
	};

	std::vector<Point> m_Points;
	XMMATRIX transform = XMMatrixIdentity();
	Color m_Color;

	XMFLOAT3 GetPosition(int n);
	XMFLOAT3 GetColor(int n);
	XMMATRIX& GetTransform();

	PolygonLight(float* points, float* color, int vnum = 3);
	void ResetPoint(float* points);
};

class PolygonLightStack
{
public:
	PolygonLightStack() = default;
	~PolygonLightStack() {}

	XMFLOAT3& GetPosition(uint i, int n);
	XMFLOAT3& GetColor(uint i, int n);
	XMMATRIX& GetTransform(uint i);
	float SetTransform(uint i, XMMATRIX& matrix);
	void ResetPoint(float* points, int i)
	{
		m_Lights[i].ResetPoint(points);
	}

	void PushPolygon(float* points, float* color, int vnum = 3);
	void UpdateDynamicVB(UploadBuffer<Geometry::Vertex>* dynamicVB);

	std::vector<uint16>& GetIndices16();
	std::vector<Vertex>& GetVertex();

	uint CountLights();
	uint CountVertex();

	bool UseTransform = true;
private:
	std::vector<uint16> mIndices16;
	std::vector<Vertex> mVertex;
	std::vector<PolygonLight> m_Lights;
};