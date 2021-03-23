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

	Microsoft::WRL::ComPtr<ID3D12Resource> ext_UploadHeap1 = nullptr;
	Microsoft::WRL::ComPtr<ID3D12Resource> ext_UploadHeap2 = nullptr;
	Microsoft::WRL::ComPtr<ID3D12Resource> ext_UploadHeap3 = nullptr;
	Microsoft::WRL::ComPtr<ID3D12Resource> ext_UploadHeap4 = nullptr;
	Microsoft::WRL::ComPtr<ID3D12Resource> ext_UploadHeap5 = nullptr;
	Microsoft::WRL::ComPtr<ID3D12Resource> ext_UploadHeap6 = nullptr;
};