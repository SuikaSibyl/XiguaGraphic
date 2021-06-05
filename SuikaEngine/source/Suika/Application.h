#pragma once

#include "Core.h"
#include "SuikaGraphics.h"
#include <QtWidgets/QApplication>

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
	};

	// To be defined in Client
	Application* CreateApplication(int argc, char* argv[]);
}
