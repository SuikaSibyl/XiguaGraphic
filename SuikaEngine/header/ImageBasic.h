#pragma once

#include <Windows.h>
#include <string>
#include <vector>
#include "../thirdparty/CImg/CImg.h"
#include "../../thirdparty/StdImage/stb_image.h"
using namespace cimg_library;
using namespace std;

namespace IMG
{
	/// <summary>
	/// Template Color with 3 components
	/// </summary>
	template<class T>
	class Color3
	{
	public:
		T R;
		T G;
		T B;
	};

	/// <summary>
	/// Template Color with 4 components
	/// </summary>
	template<class T>
	class Color4
	{
	public:
		T R;
		T G;
		T B;
		T A;
	};

	/// <summary>
	/// Record the details of the Image
	/// </summary>
	struct ImageHeader
	{
		uint32_t size;
		uint32_t height;
		uint32_t width;
		uint32_t depth;
		uint32_t mipMapCount;
		uint32_t rgbaChanelNums;
	};

	/// <summary>
	/// Image with uint8 pixels
	/// </summary>
	struct Image
	{
		ImageHeader header;
		vector<Color4<uint8_t>> pixels;
	};

	/// <summary>
	/// Image with float pixels
	/// </summary>
	struct HDRImage
	{
		ImageHeader header;
		vector<Color4<float>> pixels;
	};

	/// <summary>
	/// Cubemap Image with uint8 pixels
	/// </summary>
	struct CubemapImage
	{
		ImageHeader header;
		vector<Color4<uint8_t>> sub_pixels[6];
	};

	/// <summary>
	/// Help to read pictures in different formats.
	/// </summary>
	class ImageHelper
	{
	public:
		static void CreatePic();
		static Image ReadPic(std::wstring path, std::wstring postfix);
		static HDRImage ReadHDRPic(std::wstring path);
		static CubemapImage ReadCubemapPic(std::wstring prename, std::wstring postfix);

	private:
		// Cast wstring to string
		static string wstring2string(wstring wstr)
		{
			string result;
			// Get buffer size and imply space, size in byte
			int len = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), wstr.size(), NULL, 0, NULL, NULL);
			char* buffer = new char[len + 1];
			// Change wide char to multibyte code 
			WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), wstr.size(), buffer, len, NULL, NULL);
			buffer[len] = '\0';
			// Delete buffer and return
			result.append(buffer);
			delete[] buffer;
			return result;
		}
	};
}