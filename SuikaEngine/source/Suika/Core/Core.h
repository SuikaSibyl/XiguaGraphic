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

#ifdef SUIKA_ENABLE_ASSERTS
	#define SUIKA_ASSERT(x, ...) {if(!(x)){SUIKA_ERROR("Assertion Failed: {0}", __VA_ARGS__); __debugbreak();}}
	#define SUIKA_CORE_ASSERT(x, ...) {if(!(x)){SUIKA_CORE_ERROR("Assertion Failed: {0}", __VA_ARGS__); __debugbreak();}}
#else
	#define SUIKA_ASSERT(x, ...)
	#define SUIKA_CORE_ASSERT(x, ...)
#endif // SUIKA_ENABLE_ASSERTS

#define BIT(x) (1<<x)