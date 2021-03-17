#include <ImageBasic.h>

using namespace IMG;


void ImageHelper::ImageHelper::CreatePic()
{
	//ReadPic(L"./Resource/Textures/test.bmp");
	CImg<unsigned char>img(600, 400, 1, 3);
	img.fill(128);
	img.display("My first image");
}

Image ImageHelper::ImageHelper::ReadPic(std::wstring path)
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