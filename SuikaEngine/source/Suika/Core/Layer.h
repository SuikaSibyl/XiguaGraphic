#pragma once

#include "Core.h"
#include <Suika/Event/Event.h>

namespace Suika
{
	class SUIKA_API Layer
	{
	public:
		Layer(const std::string& name = "Layer");
		virtual ~Layer();

		virtual void OnAttach() {}
		virtual void OnDetache() {}
		virtual void OnUpadate() {}
		//virtual void OnEvent(Event& event) {}

		inline const std::string& GetName() const { return m_DebugName; }

	protected:
		std::string m_DebugName;
	};
}