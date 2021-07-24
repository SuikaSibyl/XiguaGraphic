#pragma once

#include "Core.h"
#include "LayerStack.h"
#include "SuikaGraphics.h"
#include <QtWidgets/QApplication>
#include "Window.h"

namespace Suika
{
	class SUIKA_API Application
	{
	public:
		Application(int argc, char* argv[]);
		virtual ~Application();

		void Run();

		QApplication a;
		SuikaGraphics w;


		void PushLayer(Layer* layer);
		void PushOverlay(Layer* layer);

	private:
		std::unique_ptr<Window> m_Window;
		LayerStack m_LayerStack;
	};

	// To be defined in Client
	Application* CreateApplication(int argc, char* argv[]);
}
