#pragma once

#include <Mesh.h>
#include <vector>

namespace Suika
{
	class Scene
	{
	public:
		std::vector<Suika::CudaTriangleModel*> models;

	};
}