#pragma once

#include <wrl.h>
#include <d3d11_1.h>
#include "d3dx12.h"

#pragma warning(push)
#pragma warning(disable : 4005)
#include <stdint.h>

#pragma warning(pop)

#if defined(_MSC_VER) && (_MSC_VER<1610) && !defined(_In_reads_)
#define _In_reads_(exp)
#define _Out_writes_(exp)c
#define _In_reads_bytes_(exp)
#define _In_reads_opt_(exp)
#define _Outptr_opt_
#endif

#ifndef _Use_decl_annotations_
#define _Use_decl_annotations_
#endif

namespace DirectX
{
    enum DDS_ALPHA_MODE
    {
        DDS_ALPHA_MODE_UNKNOWN = 0,
        DDS_ALPHA_MODE_STRAIGHT = 1,
        DDS_ALPHA_MODE_PREMULTIPLIED = 2,
        DDS_ALPHA_MODE_OPAQUE = 3,
        DDS_ALPHA_MODE_CUSTOM = 4,
    };

    // Standard version
    HRESULT CreateDDSTextureFromMemory(_In_ ID3D11Device* d3dDevice,
        _In_reads_bytes_(ddsDataSize) const uint8_t* ddsData,
        _In_ size_t ddsDataSize,
        _Outptr_opt_ ID3D11Resource** texture,
        _Outptr_opt_ ID3D11ShaderResourceView** textureView,
        _In_ size_t maxsize = 0,
        _Out_opt_ DDS_ALPHA_MODE* alphaMode = nullptr
    );

    HRESULT CreateDDSTextureFromMemory12(_In_ ID3D12Device* device,
        _In_ ID3D12GraphicsCommandList* cmdList,
        _In_reads_bytes_(ddsDataSize) const uint8_t* ddsData,
        _In_ size_t ddsDataSize,
        _Out_ Microsoft::WRL::ComPtr<ID3D12Resource>& texture,
        _Out_ Microsoft::WRL::ComPtr<ID3D12Resource>& textureUploadHeap,
        _In_ size_t maxsize = 0,
        _Out_opt_ DDS_ALPHA_MODE* alphaMode = nullptr
    );

    HRESULT CreateDDSTextureFromFile(_In_ ID3D11Device* d3dDevice,
        _In_z_ const wchar_t* szFileName,
        _Outptr_opt_ ID3D11Resource** texture,
        _Outptr_opt_ ID3D11ShaderResourceView** textureView,
        _In_ size_t maxsize = 0,
        _Out_opt_ DDS_ALPHA_MODE* alphaMode = nullptr
    );

    HRESULT CreateDDSTextureFromFile12(_In_ ID3D12Device* device,
        _In_ ID3D12GraphicsCommandList* cmdList,
        _In_z_ const wchar_t* szFileName,
        _Out_ Microsoft::WRL::ComPtr<ID3D12Resource>& texture,
        _Out_ Microsoft::WRL::ComPtr<ID3D12Resource>& textureUploadHeap,
        _In_ size_t maxsize = 0,
        _Out_opt_ DDS_ALPHA_MODE* alphaMode = nullptr
    );

    // Standard version with optional auto-gen mipmap support
    HRESULT CreateDDSTextureFromMemory(_In_ ID3D11Device* d3dDevice,
        _In_opt_ ID3D11DeviceContext* d3dContext,
        _In_reads_bytes_(ddsDataSize) const uint8_t* ddsData,
        _In_ size_t ddsDataSize,
        _Outptr_opt_ ID3D11Resource** texture,
        _Outptr_opt_ ID3D11ShaderResourceView** textureView,
        _In_ size_t maxsize = 0,
        _Out_opt_ DDS_ALPHA_MODE* alphaMode = nullptr
    );

    HRESULT CreateDDSTextureFromFile(_In_ ID3D11Device* d3dDevice,
        _In_opt_ ID3D11DeviceContext* d3dContext,
        _In_z_ const wchar_t* szFileName,
        _Outptr_opt_ ID3D11Resource** texture,
        _Outptr_opt_ ID3D11ShaderResourceView** textureView,
        _In_ size_t maxsize = 0,
        _Out_opt_ DDS_ALPHA_MODE* alphaMode = nullptr
    );

    // Extended version
    HRESULT CreateDDSTextureFromMemoryEx(_In_ ID3D11Device* d3dDevice,
        _In_reads_bytes_(ddsDataSize) const uint8_t* ddsData,
        _In_ size_t ddsDataSize,
        _In_ size_t maxsize,
        _In_ D3D11_USAGE usage,
        _In_ unsigned int bindFlags,
        _In_ unsigned int cpuAccessFlags,
        _In_ unsigned int miscFlags,
        _In_ bool forceSRGB,
        _Outptr_opt_ ID3D11Resource** texture,
        _Outptr_opt_ ID3D11ShaderResourceView** textureView,
        _Out_opt_ DDS_ALPHA_MODE* alphaMode = nullptr
    );

    HRESULT CreateDDSTextureFromFileEx(_In_ ID3D11Device* d3dDevice,
        _In_z_ const wchar_t* szFileName,
        _In_ size_t maxsize,
        _In_ D3D11_USAGE usage,
        _In_ unsigned int bindFlags,
        _In_ unsigned int cpuAccessFlags,
        _In_ unsigned int miscFlags,
        _In_ bool forceSRGB,
        _Outptr_opt_ ID3D11Resource** texture,
        _Outptr_opt_ ID3D11ShaderResourceView** textureView,
        _Out_opt_ DDS_ALPHA_MODE* alphaMode = nullptr
    );

    // Extended version with optional auto-gen mipmap support
    HRESULT CreateDDSTextureFromMemoryEx(_In_ ID3D11Device* d3dDevice,
        _In_opt_ ID3D11DeviceContext* d3dContext,
        _In_reads_bytes_(ddsDataSize) const uint8_t* ddsData,
        _In_ size_t ddsDataSize,
        _In_ size_t maxsize,
        _In_ D3D11_USAGE usage,
        _In_ unsigned int bindFlags,
        _In_ unsigned int cpuAccessFlags,
        _In_ unsigned int miscFlags,
        _In_ bool forceSRGB,
        _Outptr_opt_ ID3D11Resource** texture,
        _Outptr_opt_ ID3D11ShaderResourceView** textureView,
        _Out_opt_ DDS_ALPHA_MODE* alphaMode = nullptr
    );

    HRESULT CreateDDSTextureFromFileEx(_In_ ID3D11Device* d3dDevice,
        _In_opt_ ID3D11DeviceContext* d3dContext,
        _In_z_ const wchar_t* szFileName,
        _In_ size_t maxsize,
        _In_ D3D11_USAGE usage,
        _In_ unsigned int bindFlags,
        _In_ unsigned int cpuAccessFlags,
        _In_ unsigned int miscFlags,
        _In_ bool forceSRGB,
        _Outptr_opt_ ID3D11Resource** texture,
        _Outptr_opt_ ID3D11ShaderResourceView** textureView,
        _Out_opt_ DDS_ALPHA_MODE* alphaMode = nullptr
    );
}