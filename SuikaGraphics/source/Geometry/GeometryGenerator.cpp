#include <GeometryGenerator.h>

using namespace ProceduralGeometry;

GeometryGenerator::MeshData GeometryGenerator::CreateCylinder(
	float bottomRadius, float topRadius, float height, uint32 sliceCount, uint32 stackCount)
{
	MeshData meshData;
	//
	// Build Stacks.
	//
	float stackHeight = height / stackCount;
	// Amount to increment radius as we move up each stack level from bottom to top.
	float radiusStep = (topRadius - bottomRadius) / stackCount;
	uint32 ringCount = stackCount + 1;
	// Compute vertices for each stack ring starting at the bottomand moving up.
	for (uint32 i = 0; i < ringCount; ++i)
	{
		float y = -0.5f * height + i * stackHeight;
		float r = bottomRadius + i * radiusStep;// vertices of ring
		float dTheta = 2.0f * XM_PI / sliceCount;
		for (uint32 j = 0; j <= sliceCount; ++j)
		{
			Vertex vertex;
			float c = cosf(j * dTheta);
			float s = sinf(j * dTheta);
			vertex.Position = XMFLOAT3(r * c, y, r * s);
			// UV equals to unfold the  the side geometry to a rectangle
			// uvͼ���൱�ڽ�Բ��Ͱ����չ��Ϊ����
			vertex.TexC.x = (float)j / sliceCount;
			vertex.TexC.y = 1.0f - (float)i / stackCount;
			// Cylinder can be parameterized as follows, where we introduce v
			// parameter that goes in the same direction as the v tex - coord
			// so that the bitangent goes in the same direction as the
			// v tex-coord.
			// Let r0 be the bottom radius and let r1 be the top radius.
			// y(v) = h - hv for v in [0,1].
			// r(v) = r1 + (r0-r1)v
			//
			// x(t, v) = r(v)*cos(t)
			// y(t, v) = h - hv
			// z(t, v) = r(v)*sin(t)
			//
			// dx/dt = -r(v)*sin(t)
			// dy/dt = 0// dz/dt = +r(v)*cos(t)
			//
			// dx/dv = (r0-r1)*cos(t)
			// dy/dv = -h
			// dz/dv = (r0-r1)*sin(t)
			// This is unit length.
			// 
			// In tangent space,
			//  + Tangent: the direction the u axis increases
			//  + Bitangent: N X T
			//  + Normal : the direciton of the no uv increase
			// ���߿ռ��Z�� == ģ�Ͷ��㱾��ķ��߷���
			// ���ߣ�Tangent����UV�����U������룬�༴���ӷ���
			vertex.TangentU = XMFLOAT3(-s, 0.0f, c);
			float dr = bottomRadius - topRadius;
			XMFLOAT3 bitangent(dr * c, -height, dr * s);
			XMVECTOR T = XMLoadFloat3(&vertex.TangentU);
			XMVECTOR B = XMLoadFloat3(&bitangent);
			XMVECTOR N = XMVector3Normalize(XMVector3Cross(T, B));
			XMStoreFloat3(&vertex.Normal, N);
			meshData.Vertices.push_back(vertex);
		}
	}

	// Add one because we duplicate the first and last vertex per ring
	// since the texture coordinates are different.
	uint32 ringVertexCount = sliceCount + 1;
	// Compute indices for each stack.
	for (uint32 i = 0; i < stackCount; ++i)
	{
		for (uint32 j = 0; j < sliceCount; ++j)
		{
			meshData.Indices32.push_back(i * ringVertexCount +
				j);
			meshData.Indices32.push_back((i + 1) * ringVertexCount
				+ j);
			meshData.Indices32.push_back((i + 1) * ringVertexCount
				+ j + 1);
			meshData.Indices32.push_back(i * ringVertexCount +
				j);
			meshData.Indices32.push_back((i + 1) * ringVertexCount
				+ j + 1);
			meshData.Indices32.push_back(i * ringVertexCount +
				j + 1);
		}
	}
	BuildCylinderTopCap(bottomRadius, topRadius,
		height, sliceCount, stackCount, meshData);
	BuildCylinderBottomCap(bottomRadius, topRadius,
		height, sliceCount, stackCount, meshData);
	return meshData;
}

GeometryGenerator::MeshData GeometryGenerator::CreateGeosphere(float radius, uint32 numSubdivisions)
{
	MeshData meshData;
	// Put a cap on the number of subdivisions.
	numSubdivisions = std::min<uint32>(numSubdivisions,
		6u);
	// Approximate a sphere by tessellating an icosahedron.
	const float X = 0.525731f;
	const float Z = 0.850651f;
	XMFLOAT3 pos[12] =
	{
		XMFLOAT3(-X, 0.0f, Z), XMFLOAT3(X, 0.0f, Z),
		XMFLOAT3(-X, 0.0f, -Z), XMFLOAT3(X, 0.0f, -Z),
		XMFLOAT3(0.0f, Z, X), XMFLOAT3(0.0f, Z, -X),
		XMFLOAT3(0.0f, -Z, X), XMFLOAT3(0.0f, -Z, -X),
		XMFLOAT3(Z, X, 0.0f), XMFLOAT3(-Z, X, 0.0f),
		XMFLOAT3(Z, -X, 0.0f), XMFLOAT3(-Z, -X, 0.0f)
	};
	uint32 k[60] =
	{
		1,4,0, 4,9,0, 4,5,9, 8,5,4, 1,8,4,
		1,10,8, 10,3,8, 8,3,5, 3,2,5, 3,7,2,
		3,10,7, 10,6,7, 6,11,7, 6,0,11, 6,1,0,
		10,1,6, 11,0,9, 2,11,9, 5,2,9, 11,2,7
	};
	meshData.Vertices.resize(12); meshData.Indices32.assign(&k[0], &k[60]);
	for (uint32 i = 0; i < 12; ++i)
		meshData.Vertices[i].Position = pos[i];
	for (uint32 i = 0; i < numSubdivisions; ++i)
		Subdivide(meshData);
	// Project vertices onto sphere and scale.
	for (uint32 i = 0; i < meshData.Vertices.size(); ++i)
	{
		// Project onto unit sphere.
		XMVECTOR n =
			XMVector3Normalize(XMLoadFloat3(&meshData.Vertices[i].Position));
		// Project onto sphere.
		XMVECTOR p = radius * n;
		XMStoreFloat3(&meshData.Vertices[i].Position, p);
		XMStoreFloat3(&meshData.Vertices[i].Normal, n);
		// Derive texture coordinates from spherical coordinates.
		float theta = atan2f(meshData.Vertices[i].Position.z, meshData.Vertices[i].Position.x);
		// Put in [0, 2pi].
		if (theta < 0.0f)
			theta += XM_2PI;
		float phi = acosf(meshData.Vertices[i].Position.y
			/ radius); meshData.Vertices[i].TexC.x = theta / XM_2PI;
		meshData.Vertices[i].TexC.y = phi / XM_PI;
		// Partial derivative of P with respect to theta
		meshData.Vertices[i].TangentU.x = -
			radius * sinf(phi) * sinf(theta);
		meshData.Vertices[i].TangentU.y = 0.0f;
		meshData.Vertices[i].TangentU.z =
			+radius * sinf(phi) * cosf(theta);
		XMVECTOR T =
			XMLoadFloat3(&meshData.Vertices[i].TangentU);
		XMStoreFloat3(&meshData.Vertices[i].TangentU,
			XMVector3Normalize(T));
	}
	return meshData;
}

GeometryGenerator::MeshData GeometryGenerator::CreateGrid(float width, float depth, uint32 m, uint32 n)
{
	MeshData meshData;
	uint32 vertexCount = m * n;
	uint32 faceCount = (m - 1) * (n - 1) * 2;
	float halfWidth = 0.5f * width;
	float halfDepth = 0.5f * depth;
	float dx = width / (n - 1);
	float dz = depth / (m - 1);
	float du = 1.0f / (n - 1);
	float dv = 1.0f / (m - 1); meshData.Vertices.resize(vertexCount);
	for (uint32 i = 0; i < m; ++i)
	{
		float z = halfDepth - i * dz;
		for (uint32 j = 0; j < n; ++j)
		{
			float x = -halfWidth + j * dx;
			meshData.Vertices[i * n + j].Position = XMFLOAT3(x,
				0.0f, z);
			meshData.Vertices[i * n + j].Normal =
				XMFLOAT3(0.0f, 1.0f, 0.0f);
			meshData.Vertices[i * n + j].TangentU =
				XMFLOAT3(1.0f, 0.0f, 0.0f);
			// Stretch texture over grid.
			meshData.Vertices[i * n + j].TexC.x = j * du;
			meshData.Vertices[i * n + j].TexC.y = i * dv;
		}
	}

	meshData.Indices32.resize(faceCount * 3);
	// 3 indices per face
	// Iterate over each quad and compute indices.
	uint32 k = 0;
	for (uint32 i = 0; i < m - 1; ++i)
	{
		for (uint32 j = 0; j < n - 1; ++j)
		{
			meshData.Indices32[k] = i * n + j;
			meshData.Indices32[k + 1] = i * n + j + 1;
			meshData.Indices32[k + 2] = (i + 1) * n + j;
			meshData.Indices32[k + 3] = (i + 1) * n + j;
			meshData.Indices32[k + 4] = i * n + j + 1;
			meshData.Indices32[k + 5] = (i + 1) * n + j + 1;
			k += 6; // next quad
		}
	}

	return meshData;
}

void GeometryGenerator::BuildCylinderTopCap(
	float bottomRadius, float topRadius, float height,
	uint32 sliceCount, uint32 stackCount, MeshData&
	meshData)
{
	uint32 baseIndex = (uint32)meshData.Vertices.size();
	float y = 0.5f * height;
	float dTheta = 2.0f * XM_PI / sliceCount;
	// Duplicate cap ring vertices because the texture coordinatesand
	// normals differ.
	for (uint32 i = 0; i <= sliceCount; ++i)
	{
		float x = topRadius * cosf(i * dTheta);
		float z = topRadius * sinf(i * dTheta);
		// Scale down by the height to try and make top cap texture coord
		// area proportional to base.
		float u = x / height + 0.5f;
		float v = z / height + 0.5f;
		meshData.Vertices.push_back(
			Vertex(x, y, z, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
				0.0f, u, v));
	}
	// Cap center vertex.
	meshData.Vertices.push_back(
		Vertex(0.0f, y, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
			0.0f, 0.0f, 0.5f, 0.5f));// Index of center vertex.
	uint32 centerIndex =
		(uint32)meshData.Vertices.size() - 1;
	for (uint32 i = 0; i < sliceCount; ++i)
	{
		meshData.Indices32.push_back(centerIndex);
		meshData.Indices32.push_back(baseIndex + i + 1);
		meshData.Indices32.push_back(baseIndex + i);
	}
}

void GeometryGenerator::BuildCylinderBottomCap(
	float bottomRadius, float topRadius, float height,
	uint32 sliceCount, uint32 stackCount, MeshData&
	meshData)
{
	uint32 baseIndex = (uint32)meshData.Vertices.size();
	float y = -0.5f * height;
	float dTheta = 2.0f * XM_PI / sliceCount;
	// Duplicate cap ring vertices because the texture coordinatesand
	// normals differ.
	for (uint32 i = 0; i <= sliceCount; ++i)
	{
		float x = bottomRadius * cosf(i * dTheta);
		float z = bottomRadius * sinf(i * dTheta);
		// Scale down by the height to try and make top cap texture coord
		// area proportional to base.
		float u = x / height + 0.5f;
		float v = 0.5f - z / height;
		meshData.Vertices.push_back(
			Vertex(x, y, z, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,
				0.0f, u, v));
	}
	// Cap center vertex.
	meshData.Vertices.push_back(
		Vertex(0.0f, y, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f,
			0.0f, 0.0f, 0.5f, 0.5f));// Index of center vertex.
	uint32 centerIndex =
		(uint32)meshData.Vertices.size() - 1;
	for (uint32 i = 0; i < sliceCount; ++i)
	{
		meshData.Indices32.push_back(centerIndex);
		meshData.Indices32.push_back(baseIndex + i);
		meshData.Indices32.push_back(baseIndex + i + 1);
	}
}

void GeometryGenerator::Subdivide(MeshData& meshData)
{
	// Save a copy of the input geometry.
	MeshData inputCopy = meshData;


	meshData.Vertices.resize(0);
	meshData.Indices32.resize(0);

	//       v1
	//       *
	//      / \
	//     /   \
	//  m0*-----*m1
	//   / \   / \
	//  /   \ /   \
	// *-----*-----*
	// v0    m2     v2

	UINT numTris = inputCopy.Indices32.size() / 3;
	for (UINT i = 0; i < numTris; ++i)
	{
		Vertex v0 = inputCopy.Vertices[inputCopy.Indices32[i * 3 + 0]];
		Vertex v1 = inputCopy.Vertices[inputCopy.Indices32[i * 3 + 1]];
		Vertex v2 = inputCopy.Vertices[inputCopy.Indices32[i * 3 + 2]];

		//
		// Generate the midpoints.
		//

		Vertex m0, m1, m2;

		// For subdivision, we just care about the position component.  We derive the other
		// vertex components in CreateGeosphere.

		m0.Position = XMFLOAT3(
			0.5f * (v0.Position.x + v1.Position.x),
			0.5f * (v0.Position.y + v1.Position.y),
			0.5f * (v0.Position.z + v1.Position.z));

		m1.Position = XMFLOAT3(
			0.5f * (v1.Position.x + v2.Position.x),
			0.5f * (v1.Position.y + v2.Position.y),
			0.5f * (v1.Position.z + v2.Position.z));

		m2.Position = XMFLOAT3(
			0.5f * (v0.Position.x + v2.Position.x),
			0.5f * (v0.Position.y + v2.Position.y),
			0.5f * (v0.Position.z + v2.Position.z));

		//
		// Add new geometry.
		//

		meshData.Vertices.push_back(v0); // 0
		meshData.Vertices.push_back(v1); // 1
		meshData.Vertices.push_back(v2); // 2
		meshData.Vertices.push_back(m0); // 3
		meshData.Vertices.push_back(m1); // 4
		meshData.Vertices.push_back(m2); // 5

		meshData.Indices32.push_back(i * 6 + 0);
		meshData.Indices32.push_back(i * 6 + 3);
		meshData.Indices32.push_back(i * 6 + 5);

		meshData.Indices32.push_back(i * 6 + 3);
		meshData.Indices32.push_back(i * 6 + 4);
		meshData.Indices32.push_back(i * 6 + 5);

		meshData.Indices32.push_back(i * 6 + 5);
		meshData.Indices32.push_back(i * 6 + 4);
		meshData.Indices32.push_back(i * 6 + 2);

		meshData.Indices32.push_back(i * 6 + 3);
		meshData.Indices32.push_back(i * 6 + 1);
		meshData.Indices32.push_back(i * 6 + 4);
	}
}
