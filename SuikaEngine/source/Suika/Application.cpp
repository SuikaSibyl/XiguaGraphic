#include "Application.h"


namespace Suika
{
	Application::Application(int argc, char* argv[]) :a(argc, argv)
	{

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
}