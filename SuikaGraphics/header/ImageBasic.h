#pragma once

#include <Windows.h>
#include <string>
#include "../thirdparty/CImg/CImg.h"
using namespace cimg_library;
using namespace std;

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

class ImageHelper
{
public:
	static void CreatePic();
	static Image ReadPic(std::wstring path);

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

void ImageHelper::CreatePic()
{
	//ReadPic(L"./Resource/Textures/test.bmp");
	CImg<unsigned char>img(600, 400, 1, 3);
	img.fill(128);
	img.display("My first image");
}

Image ImageHelper::ReadPic(std::wstring path)
{
	CImg<uint8_t> ReadTexture(wstring2string(path).c_str());
	ReadTexture.display("My first image");

	Image res;
	res.header.height = ReadTexture.height();
	res.header.width = ReadTexture.width();
	res.header.depth = ReadTexture.depth();
	res.header.mipMapCount = 0;
	res.header.rgbaChanelNums = ReadTexture.spectrum();
	res.header.size = ReadTexture.size();

	vector<Color3<uint8_t>> ReadTextureBulkData;
	vector<Color4<uint8_t>> InitTextureBulkData;

	cimg_forXY(ReadTexture, x, y)
	{
		Color3<uint8_t> NewColor;
		NewColor.R = ReadTexture(x, y, 0);
		NewColor.G = ReadTexture(x, y, 1);
		NewColor.B = ReadTexture(x, y, 2);
		ReadTextureBulkData.push_back(std::move(NewColor));
	}

	for (int i = 0; i < ReadTextureBulkData.size(); i++)
	{
		Color4<uint8_t> NewColor;
		NewColor.R = ReadTextureBulkData[i].R;
		NewColor.G = ReadTextureBulkData[i].G;
		NewColor.B = ReadTextureBulkData[i].B;
		NewColor.A = 255;
		InitTextureBulkData.push_back(NewColor);
	}

	res.pixels = std::move(InitTextureBulkData);
	return  std::move(res);
}