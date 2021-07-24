#include <Precompiled.h>
#include "WindowsWindow.h"

namespace Suika
{
	static bool s_Initialized = false;

	Window* Window::Create(const WindowProps& props)
	{
		return new WindowsWindow(props);
	}

	WindowsWindow::WindowsWindow(const WindowProps& props)
	{
		Init(props);
	}

	WindowsWindow::~WindowsWindow()
	{
		Shutdown();
	}

	void WindowsWindow::Init(const WindowProps& props)
	{
		m_Data.Title = props.Title;
		m_Data.Width = props.Width;
		m_Data.Height = props.Height;

		SUIKA_CORE_INFO("Creating window {0} ({1}, {2})", props.Title, props.Width, props.Height);

		if (!s_Initialized)
		{

			s_Initialized = true;
		}
	}

	void WindowsWindow::OnUpdate()
	{

	}

	// Window attriutes
	inline void WindowsWindow::SetEventCallback(const EventCallbackFn& callback)
	{

	}

	void WindowsWindow::SetVSync(bool enabled)
	{

	}

	bool WindowsWindow::IsVSync() const
	{
		return false;
	}

	void WindowsWindow::Shutdown()
	{
		// Destroy the window

	}

}