#pragma once

#include <unordered_map>
#include <string>
#include <QWidget>
#include <QEvent>
#include <QWheelEvent>
#include <vector>
#include <Debug.h>

using std::vector;
using std::string;

class InputSystem
{
public:
	enum InputTypes
	{
		Forward,
		Back,
		Left,
		Right,
	};

	InputSystem()
	{
		// Init Keyboard Pressed
		for each (InputTypes type in m_InputTypes)
		{
			KeyboardPressed.insert(std::pair<InputTypes, bool>(type, false));
		}
		// Init MapKey2InputType
		for each (std::pair<int, InputTypes> pair in m_Key2InputType)
		{
			MapKey2InputType.insert(std::pair<int, InputTypes>(pair.first, pair.second));
		}
	}

public:
	std::unordered_map<InputTypes, bool> KeyboardPressed;

	void KeyPressed(QKeyEvent* event)
	{
		Debug::Log("KeyPressed");
		KeyboardPressed[MapKey2InputType[event->key()]] = true;
	}

	void KeyReleased(QKeyEvent* event)
	{
		Debug::LogError("KeyRelease");
		KeyboardPressed[MapKey2InputType[event->key()]] = true;
	}

private:
	std::unordered_map<int, InputTypes> MapKey2InputType;

	vector<InputTypes> m_InputTypes{
		Forward, 
		Back, 
		Left, 
		Right
	};

	vector<std::pair<int, InputTypes>> m_Key2InputType{
		{Qt::Key_W, Forward},
		{Qt::Key_S, Back},
		{Qt::Key_A, Left},
		{Qt::Key_D, Right},
	};
};