#pragma once

#ifdef SUIKA_PLATFORM_WINDOWS
	#ifdef SUIKA_BUILD_DLL
		#define SUIKA_API _declspec(dllexport)
	#else
		#define SUIKA_API _declspec(dllimport)
	#endif // SUIKA_BUILD_DLL
#else
	#error SUIKA only support Windows!
#endif

#define BIT(x) (1<<x)