#pragma once

#include <entt.hpp>

namespace SUIKA
{
	class Scene
	{
	public:
		Scene();
		~Scene();
	private:
		entt::registry m_Registery;

	};
}