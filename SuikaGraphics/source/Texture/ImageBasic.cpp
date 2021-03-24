#include <ImageBasic.h>

using namespace IMG;


void ImageHelper::ImageHelper::CreatePic()
{
	//ReadPic(L"./Resource/Textures/test.bmp");
	CImg<unsigned char>img(600, 400, 1, 3);
	img.fill(128);
	img.display("My first image");
}

Image ImageHelper::ImageHelper::ReadPic(std::wstring path, std::wstring postfix)
{
	if (postfix == L"jpg" || postfix == L"png")
	{
		stbi_set_flip_vertically_on_load(true);
		int width, height, nrComponents;
		unsigned char* data = stbi_load(wstring2string(path).c_str(), &width, &height, &nrComponents, 3);

		if (data)
		{
			Image res;
			res.header.height = height;
			res.header.width = width;
			res.header.depth = 1;
			res.header.mipMapCount = 0;
			res.header.rgbaChanelNums = nrComponents;
			res.header.size = height * width * nrComponents;

			vector<Color4<uint8_t>> InitTextureBulkData;

			for (uint32_t h = 0; h < width; h++)
			{
				for (uint32_t w = 0; w < height; w++)
				{
					Color4<uint8_t> NewColor;
					uint32_t start = 3 * height * h + 3 * w;
					NewColor.R = data[start + 0];
					NewColor.G = data[start + 1];
					NewColor.B = data[start + 2];
					NewColor.A = 255;
					InitTextureBulkData.push_back(std::move(NewColor));
				}
			}

			res.pixels = std::move(InitTextureBulkData);
			stbi_image_free(data);
			return  std::move(res);
		}
	}
	else
	{
		CImg<uint8_t> ReadTexture(wstring2string(path).c_str());

		Image res;
		res.header.height = ReadTexture.height();
		res.header.width = ReadTexture.width();
		res.header.depth = ReadTexture.depth();
		res.header.mipMapCount = 0;
		res.header.rgbaChanelNums = ReadTexture.spectrum();
		res.header.size = ReadTexture.size();

		vector<Color4<uint8_t>> InitTextureBulkData;

		cimg_forXY(ReadTexture, x, y)
		{
			Color4<uint8_t> NewColor;
			NewColor.R = ReadTexture(x, y, 0);
			NewColor.G = ReadTexture(x, y, 1);
			NewColor.B = ReadTexture(x, y, 2);
			NewColor.A = 255;
			InitTextureBulkData.push_back(NewColor);
		}

		res.pixels = std::move(InitTextureBulkData);
		return  std::move(res);
	}
}

CubemapImage ImageHelper::ReadCubemapPic(std::wstring prename, std::wstring postfix)
{
	// Load all six pictures
	bool load_success = true;
	stbi_set_flip_vertically_on_load(false);
	int width, height, nrComponents;
	unsigned char* datas[6];
	wstring index[6] = { L"1", L"2", L"3", L"4", L"5", L"6" };
	for (int i = 0; i < 6; i++)
	{
		std::wstring path = prename + L"_" + index[i] + L"." + postfix;
		datas[i] = stbi_load(wstring2string(path).c_str(), &width, &height, &nrComponents, 3);
		load_success &= (datas[i] != nullptr);
	}

	CubemapImage res;

	if (load_success)
	{
		res.header.height = height;
		res.header.width = width;
		res.header.depth = 1;
		res.header.mipMapCount = 0;
		res.header.rgbaChanelNums = nrComponents;
		res.header.size = height * width * nrComponents;

		for (int i = 0; i < 6; i++)
		{
			vector<Color4<uint8_t>> InitTextureBulkData;
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					Color4<uint8_t> NewColor;
					NewColor.R = (datas[i])[3 * width * h + 3 * w + 0];
					NewColor.G = (datas[i])[3 * width * h + 3 * w + 1];
					NewColor.B = (datas[i])[3 * width * h + 3 * w + 2];
					NewColor.A = 255;
					InitTextureBulkData.push_back(std::move(NewColor));
				}
			}
			stbi_image_free(datas[i]);
			res.sub_pixels[i] = std::move(InitTextureBulkData);
		}
		return  std::move(res);
	}

	return res;
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

		vector<Color4<float>> InitTextureBulkData;

		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				Color4<float> NewColor;
				NewColor.R = data[3 * width * h + 3 * w + 0];
				NewColor.G = data[3 * width * h + 3 * w + 1];
				NewColor.B = data[3 * width * h + 3 * w + 2];
				NewColor.A = 1;
				InitTextureBulkData.push_back(std::move(NewColor));
			}
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