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

HDRImage ImageHelper::ReadHDRPic(std::wstring path)
{
	stbi_set_flip_vertically_on_load(true);
	int width, height, nrComponents;
	float* data = stbi_loadf(wstring2string(path).c_str(), &width, &height, &nrComponents, 0);

	if (data)
	{

		HDRImage res;
		res.header.height = height;
		res.header.width = width;
		res.header.depth = 1;
		res.header.mipMapCount = 0;
		res.header.rgbaChanelNums = nrComponents;
		res.header.size = height * width * nrComponents;

		vector<Color3<float>> ReadTextureBulkData;
		vector<Color4<float>> InitTextureBulkData;

		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				Color3<float> NewColor;
				NewColor.R = data[3 * width * h + 3 * w + 0];
				NewColor.G = data[3 * width * h + 3 * w + 1];
				NewColor.B = data[3 * width * h + 3 * w + 2];
				ReadTextureBulkData.push_back(std::move(NewColor));
			}
		}

		for (int i = 0; i < ReadTextureBulkData.size(); i++)
		{
			Color4<float> NewColor;
			NewColor.R = ReadTextureBulkData[i].R;
			NewColor.G = ReadTextureBulkData[i].G;
			NewColor.B = ReadTextureBulkData[i].B;
			NewColor.A = 1;
			InitTextureBulkData.push_back(NewColor);
		}

		res.pixels = std::move(InitTextureBulkData);
		stbi_image_free(data);
		return  std::move(res);
	}
	else
	{
		// Problem
	}
}