workspace "Suika"
    architecture "x64"

    configurations
    {
        "Debug",
        "Release",
        "Dist"
    }

outputdir = "%{cfg.buildcfg}=%{cfg.system}-%{cfg.architecture}"

project "SuikaEngine"
    location "SuikaEngine"
    kind "SharedLib"
    language "C++"

    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

    files
    {
        "%{prj.name}/src/**.h",
        "%{prj.name}/src/**.cpp"
    }

    includedirs
    {
        "%{prj.name}/vendor/spdlog/include"
    }

    filter "system:windows"
        cppdialect "C++17"
        staticruntime "On"
        systemversion "latest"
        
        defines
        {
            "SUIKA_PLATFORM_WINDOWS",
            "SUIKA_BUILD_DLL"
        }

        postbuildcommands
        {
            ("{COPY} %{cfg.buildtarget.relpath} ../bin/" .. outputdir .. "/Sandbox")
        }

    filter "configurations:Debug"
        defines "SUIKA_DEBUG"
        symbols "On"
        
    filter "configurations:Release"
        defines "SUIKA_RELEASE"
        optimize "On"
    
    filter "configurations:Dist"
        defines "SUIKA_DIST"
        optimize "On"

        
project "SandBox"
    location "SandBox"
    kind "ConsoleApp"
    language "C++"

    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

    files
    {
        "%{prj.name}/src/**.h",
        "%{prj.name}/src/**.cpp"
    }

    includedirs
    {
       "%{prj.name}/vendor/spdlog/include",
       "SuikaEngine/src"
    }

    links
    {
        "SuikaEngine"
    }

    filter "system:windows"
        cppdialect "C++17"
        staticruntime "On"
        systemversion "latest"
        
        defines
        {
            "SUIKA_PLATFORM_WINDOWS",
            "SUIKA_BUILD_DLL"
        }

        postbuildcommands
        {
            ("{COPY} %{cfg.buildtarget.relpath} ../bin/" .. outputdir .. "/Sandbox")
        }

    filter "configurations:Debug"
        defines "SUIKA_DEBUG"
        symbols "On"
        
    filter "configurations:Release"
        defines "SUIKA_RELEASE"
        optimize "On"
    
    filter "configurations:Dist"
        defines "SUIKA_DIST"
        optimize "On"