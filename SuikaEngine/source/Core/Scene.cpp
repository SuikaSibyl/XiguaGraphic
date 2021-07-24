#include <Precompiled.h>
#include "Scene.h"

namespace SUIKA
{
	static void DoMath(const XMMATRIX & transform)
	{

	}

	Scene::Scene()
	{

		struct TransformComponent
		{
			XMMATRIX Transform; 

			// Constructor
			TransformComponent() = default;
			TransformComponent(const TransformComponent&) = default;
			TransformComponent(const XMMATRIX& transform) : Transform(transform) {}

			operator XMMATRIX& () { return Transform; }
			operator const XMMATRIX& () const { return Transform; }
		};

		//entt::entity entity = m_Registery.create(); 
		//m_Registery.emplace<TransformComponent>(entity, XMMATRIX());

		//if (m_Registery.has<TransformComponent>(entity));
		//TransformComponent& transform = m_Registery.get<TransformComponent>(entity);
		//DoMath(transform);
		
	}

	Scene::~Scene()
	{

	}
}