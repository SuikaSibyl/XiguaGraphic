#pragma once

#include <string>
#include <Utility.h>
#include <DDSTextureLoader.h>
#include "CudaUtil.h"

struct Texture
{
	enum Type
	{
		Texture2D,
		Cubemap,
		CubemapArray,
	};

	// Unique material name for lookup.
	std::string Name;

	std::wstring Filename;

	UINT SrvIndex = -1;
	bool isCubeMap = false;
	bool isCubeMapArray = false;
	Microsoft::WRL::ComPtr<ID3D12Resource> Resource = nullptr;
	Microsoft::WRL::ComPtr<ID3D12Resource> UploadHeap = nullptr;

	UINT Channels = 0;
	UINT Width = 0;
	UINT Height = 0;
	UINT pixelBufferSize = 0;

	bool bindCuda = false;
	cudaSurfaceObject_t cuSurface;
	cudaTextureObject_t cuTexture;

	void DisposeUploaders()
	{
		UploadHeap = nullptr;
	}

	Type type;
};

class WritableTexture
{
public:
	enum WritableType
	{
		RenderTarget,
		DepthStencil,
		UnorderedAccess,
		CudaShared,
	};
public:
	WritableTexture(ID3D12Device* device, UINT width, UINT height, WritableType type);

	WritableTexture(const WritableTexture& rhs) = delete;
	WritableTexture& operator=(const WritableTexture& rhs) = delete;
	~WritableTexture() = default;

	UINT Width() const;
	UINT Height() const;

	ID3D12Resource* Resource();
	CD3DX12_CPU_DESCRIPTOR_HANDLE Srv() const;
	CD3DX12_CPU_DESCRIPTOR_HANDLE Rtv() const;
	CD3DX12_CPU_DESCRIPTOR_HANDLE Dsv() const;

	CD3DX12_GPU_DESCRIPTOR_HANDLE gpuSrv() const;
	CD3DX12_GPU_DESCRIPTOR_HANDLE gpuUav() const;
	cudaSurfaceObject_t cuSurface;
	cudaSurfaceObject_t cuTexture;

	D3D12_VIEWPORT Viewport()const;
	D3D12_RECT ScissorRect()const;

	void CreateSrvDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE hCpuSrv, CD3DX12_GPU_DESCRIPTOR_HANDLE hGpuSrv);
	void CreateRtvDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE hCpuRtv);
	void CreateDsvDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE hCpuDsv);
	void CreateUavDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE hCpuUav, CD3DX12_GPU_DESCRIPTOR_HANDLE hGpuUav);

	void OnResize(UINT newWidth, UINT newHeight);

	void CaptureTexture(const wchar_t* name, const GUID& format, ID3D12CommandQueue* ptrCmdQueue);

	int SrvIndex = 0;
	int UavIndex = 0;

	void ChangeResourceState(ID3D12GraphicsCommandList* ptr_CommandList, D3D12_RESOURCE_STATES NextState);

private:
	void BuildDescriptors();
	void BuildResource();

private:
	ID3D12Device* md3dDevice = nullptr;

	// Buffer Descriptor Handles
	CD3DX12_CPU_DESCRIPTOR_HANDLE mhCpuSrv;
	CD3DX12_CPU_DESCRIPTOR_HANDLE mhCpuRtv;
	CD3DX12_CPU_DESCRIPTOR_HANDLE mhCpuDsv;
	CD3DX12_CPU_DESCRIPTOR_HANDLE mhCpuUav;

	CD3DX12_GPU_DESCRIPTOR_HANDLE mhGpuSrv;
	CD3DX12_GPU_DESCRIPTOR_HANDLE mhGpuUav;

	// Buffer Attributes Variances
	UINT mWidth = 0;
	UINT mHeight = 0;
	D3D12_VIEWPORT mViewport;
	D3D12_RECT mScissorRect;

	// Buffer Type Variances
	WritableType textureType;
	DXGI_FORMAT mFormat = DXGI_FORMAT_UNKNOWN;
	DXGI_FORMAT mSRFormat = DXGI_FORMAT_UNKNOWN;
	DXGI_FORMAT mClearFormat = DXGI_FORMAT_UNKNOWN;
	D3D12_HEAP_FLAGS mHeapFlag = D3D12_HEAP_FLAG_NONE;
	D3D12_RESOURCE_FLAGS mFlag = D3D12_RESOURCE_FLAG_NONE;

	// Buffer Clear Information
	const float ClearColor[4] = { 0.117,0.117,0.117,1 };
	
	// Buffer Resource
	Microsoft::WRL::ComPtr<ID3D12Resource> mWritableTexture = nullptr;

	// Curr State
	D3D12_RESOURCE_STATES CurrState = D3D12_RESOURCE_STATE_COMMON;
};