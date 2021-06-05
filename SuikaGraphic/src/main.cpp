#include <iostream>
#include <Suika.h>

class Sandbox : public Suika::Application
{
public:
	Sandbox(int argc, char* argv[]) :Application(argc, argv)
	{

	}

	~Sandbox()
	{

	}
};

Suika::Application* Suika::CreateApplication(int argc, char* argv[])
{
	return new Sandbox(argc, argv);
}