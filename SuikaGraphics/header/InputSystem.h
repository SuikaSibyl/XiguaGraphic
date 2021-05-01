#pragma once

#include <unordered_map>
#include <string>
#include <QWidget>
#include <QEvent>
#include <QWheelEvent>
#include <vector>
#include <Debug.h>

#include <Delegate.h>

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
		Up,
		Down,
		Pause,
		RTRender,
		PrtScreen,
	};

	InputSystem()
	{
		// Init Keyboard Pressed
		for each (InputTypes type in m_InputTypes)
		{
			KeyboardPressed.insert(std::pair<InputTypes, bool>(type, false));
			MapKey2Delegate[type] = new Delegate::CMultiDelegate<void>();
		};
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
		InputTypes inputType = MapKey2InputType[event->key()];
		KeyboardPressed[inputType] = true;
		(*MapKey2Delegate[inputType])();
	}

	void KeyReleased(QKeyEvent* event)
	{
		KeyboardPressed[MapKey2InputType[event->key()]] = false;
	}

	template< typename T>
	void AddListening(InputTypes type, T func);

	template< typename T, typename F>
	void AddListeningMem(InputTypes type, T* _object, F func)
	{
		(*(MapKey2Delegate[type])) += Delegate::newDelegate(_object, func);
	}

private:
	std::unordered_map<int, InputTypes> MapKey2InputType;
	std::unordered_map<InputTypes, Delegate::CMultiDelegate<void>*> MapKey2Delegate;
	typedef std::unordered_map<InputTypes, Delegate::CMultiDelegate<void>*>::iterator DelegateIter;

	vector<InputTypes> m_InputTypes{
		Forward, 
		Back, 
		Left, 
		Right,
		Up,
		Down,
		Pause,
		RTRender,
		PrtScreen,
	};

	vector<std::pair<int, InputTypes>> m_Key2InputType{
		{Qt::Key_W, Forward},
		{Qt::Key_S, Back},
		{Qt::Key_A, Left},
		{Qt::Key_D, Right},
		{Qt::Key_Q, Down},
		{Qt::Key_E, Up},
		{Qt::Key_Escape, Pause},
		{Qt::Key_Space, RTRender},
		{Qt::Key_F1, PrtScreen},
	};
};