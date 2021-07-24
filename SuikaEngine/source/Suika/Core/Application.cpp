#include <Precompiled.h>
#include "Application.h"


namespace Suika
{
	Application::Application(int argc, char* argv[]) :a(argc, argv)
	{
		m_Window = std::unique_ptr<Window>(Window::Create());
	}

	Application::~Application()
	{

	}

	void Application::Run()
	{
		w.show();
		a.exec();
		return;
	}

	void Application::PushLayer(Layer* layer)
	{
		m_LayerStack.PushLayer(layer);
	}

	void Application::PushOverlay(Layer* layer)
	{
		m_LayerStack.PushOverlay(layer);
	}
}