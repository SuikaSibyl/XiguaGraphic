#pragma once

#include <Suika/Core/Window.h>
#include <Suika/Core/Log.h>

namespace Suika
{
	class WindowsWindow :public Window
	{
	public:
		WindowsWindow(const WindowProps& props);
		virtual ~WindowsWindow();

		void OnUpdate() override;

		inline unsigned int GetWidth() const override { return m_Data.Width; }
		inline unsigned int GetHeight() const override { return m_Data.Height; }

		// Window attriutes
		inline void SetEventCallback(const EventCallbackFn& callback) override;
		void SetVSync(bool enabled) override;
		bool IsVSync() const override;

	private:
		virtual void Init(const WindowProps& props);
		virtual void Shutdown();

	private:
		
		struct WindowData
		{
			std::string Title;
			unsigned int Width, Height;


		};

		WindowData m_Data;
	};
}