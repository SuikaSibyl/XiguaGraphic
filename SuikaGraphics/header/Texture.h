#pragma once

#include <string>
#include <Utility.h>
#include <DDSTextureLoader.h>

struct Texture
{
	// Unique material name for lookup.
	std::string Name;

	std::wstring Filename;

	UINT Index = -1;
	bool isCubeMap = false;
	Microsoft::WRL::ComPtr<ID3D12Resource> Resource = nullptr;
	Microsoft::WRL::ComPtr<ID3D12Resource> UploadHeap = nullptr;
};