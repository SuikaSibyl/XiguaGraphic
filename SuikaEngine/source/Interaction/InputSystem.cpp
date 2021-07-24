#include <Precompiled.h>
#include <InputSystem.h>
#include <Camera.h>

template< typename T>
void InputSystem::AddListening(InputTypes type, T func)
{
	(*(MapKey2Delegate[type])) += Delegate::newDelegate(func);
}