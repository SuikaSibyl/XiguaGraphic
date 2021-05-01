#pragma once

#include <string>
#include <Utility.h>
#include <DDSTextureLoader.h>
#include "CudaUtil.h"

struct Texture
{
	// Unique material name for lookup.
	std::string Name;

	std::wstring Filename;

	UINT Index = -1;
	bool isCubeMap = false;
	Microsoft::WRL::ComPtr<ID3D12Resource> Resource = nullptr;
	Microsoft::WRL::ComPtr<ID3D12Resource> UploadHeap = nullptr;

	UINT Channels = 0;
	UINT Width = 0;
	UINT Height = 0;
	UINT pixelBufferSize = 0;

	bool bindCuda = false;
	cudaSurfaceObject_t cuSurface;


	void DisposeUploaders()
	{
		UploadHeap = nullptr;
	}
};

class WritableTexture
{
public:
	WritableTexture();

	WritableTexture(const WritableTexture& rhs) = delete;
	WritableTexture& operator=(const WritableTexture& rhs) = delete;
	~WritableTexture() = default;

	UINT Width() const;
	UINT Height() const;

};