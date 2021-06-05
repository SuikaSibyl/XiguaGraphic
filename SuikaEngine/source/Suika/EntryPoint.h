#pragma once

#ifdef SUIKA_PLATFORM_WINDOWS

extern Suika::Application* Suika::CreateApplication(int argc, char* argv[]);

int main(int argc, char** argv)
{
	Suika::Log::Init();
	SUIKA_CORE_WARN("Initialized Log!");
	int a = 5;
	SUIKA_APP_INFO("Hello Var={0}", a);

	auto app = Suika::CreateApplication(argc, argv);
	app->Run();
	delete app;
}

#endif // SUIKA_PLATFORM_WINDOWS
