#pragma once

#include "Core.h"
#include "spdlog/spdlog.h"

namespace Suika
{
	class SUIKA_API Log
	{
	public:
		static void Init();

		inline static std::shared_ptr<spdlog::logger>& GetCoreLogger() { return s_CoreLogger; }
		inline static std::shared_ptr<spdlog::logger>& GetClientLogger() { return s_ClientLogger; }

	private:
		static std::shared_ptr<spdlog::logger> s_CoreLogger;
		static std::shared_ptr<spdlog::logger> s_ClientLogger;
	};
}


#define SUIKA_CORE_TRACE(...)	::Suika::Log::GetCoreLogger()->trace(__VA_ARGS__)
#define SUIKA_CORE_INFO(...)	::Suika::Log::GetCoreLogger()->info(__VA_ARGS__)
#define SUIKA_CORE_WARN(...)	::Suika::Log::GetCoreLogger()->warn(__VA_ARGS__)
#define SUIKA_CORE_ERROR(...)	::Suika::Log::GetCoreLogger()->error(__VA_ARGS__)
#define SUIKA_CORE_FATAL(...)	::Suika::Log::GetCoreLogger()->fatal(__VA_ARGS__)

#define SUIKA_APP_TRACE(...)	::Suika::Log::GetClientLogger()->trace(__VA_ARGS__)
#define SUIKA_APP_INFO(...)		::Suika::Log::GetClientLogger()->info(__VA_ARGS__)
#define SUIKA_APP_WARN(...)		::Suika::Log::GetClientLogger()->warn(__VA_ARGS__)
#define SUIKA_APP_ERROR(...)	::Suika::Log::GetClientLogger()->error(__VA_ARGS__)
#define SUIKA_APP_FATAL(...)	::Suika::Log::GetClientLogger()->fatal(__VA_ARGS__)


