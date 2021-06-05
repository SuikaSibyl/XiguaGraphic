#pragma once

#include "Event.h"

#include <sstream>

namespace Suika
{
	class SUIKA_API KeyEvent : public Event
	{
	public:
		inline int GetKeyCode() const { return m_KeyCode; }

		EVENT_CLASS_CATEGORY(EventCategoryKeyboard | EventCategoryInput)
	protected:
		KeyEvent(int keycode)
			:m_KeyCode(keycode) {}

		int m_KeyCode;
	};

	class SUIKA_API KeyPressedEvent :public KeyEvent
	{
	public:
		//KeyPressedEvent()
		//{

		//}


	private:
		int m_RepeatedCount;
	};

	class SUIKA_API KeyReleasedEvent :public KeyEvent
	{
	public:
		KeyReleasedEvent(int keycode)
			:KeyEvent(keycode) {}

		std::string ToString() const override
		{
			std::stringstream ss;
			ss << "KeyReleasedEvent: " << m_KeyCode;
			return ss.str();
		}

		//EVENT_CLASS_TYPE(KeyReleased);
	};
}