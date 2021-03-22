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
	template<class T>
	class Color3
	{
	public:
		T R;
		T G;
		T B;
	};

	template<class T>
	class Color4
	{
	public:
		T R;
		T G;
		T B;
		T A;
	};

	struct ImageHeader
	{
		uint32_t size;
		uint32_t height;
		uint32_t width;
		uint32_t depth;
		uint32_t mipMapCount;
		uint32_t rgbaChanelNums;
	};

	struct Image
	{
		ImageHeader header;
		vector<Color4<uint8_t>> pixels;
	};

	struct HDRImage
	{
		ImageHeader header;
		vector<Color4<float>> pixels;
	};

	class ImageHelper
	{
	public:
		static void CreatePic();
		static Image ReadPic(std::wstring path);
		static HDRImage ReadHDRPic(std::wstring path);

	private:

		static string wstring2string(wstring wstr)
		{
			string result;
			//获取缓冲区大小，并申请空间，缓冲区大小事按字节计算的  
			int len = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), wstr.size(), NULL, 0, NULL, NULL);
			char* buffer = new char[len + 1];
			//宽字节编码转换成多字节编码  
			WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), wstr.size(), buffer, len, NULL, NULL);
			buffer[len] = '\0';
			//删除缓冲区并返回值  
			result.append(buffer);
			delete[] buffer;
			return result;
		}
	};
}