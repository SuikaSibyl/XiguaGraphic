#include <Texture.h>
#include <ScreenGrab.h>
#include <wincodec.h>

WritableTexture::WritableTexture(ID3D12Device* device, UINT width, UINT height, WritableType type):
	textureType(type)
{
	md3dDevice = device;

	mWidth = width;
	mHeight = height;

	mViewport = { 0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f };
	mScissorRect = { 0, 0, (int)width, (int)height };

	switch (textureType)
	{
	case WritableTexture::RenderTarget:
		// Is render target
		mFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
		mSRFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
		mClearFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
		mFlag = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
		break;
	case WritableTexture::DepthStencil:
		// Is depth stencil
		mFormat = DXGI_FORMAT_R24G8_TYPELESS;
		mSRFormat = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
		mClearFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
		mFlag = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
		break;
	case WritableTexture::UnorderedAccess:
		// Is unordered access texture
		mFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
		mSRFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
		mClearFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
		mFlag = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
	case WritableTexture::CudaShared:
		// Is cuda shared texture
		mFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
		mSRFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
		mClearFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
		mFlag = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		mHeapFlag = D3D12_HEAP_FLAG_SHARED;
		textureType = UnorderedAccess;
		break;
	default:
		break;
	}

	BuildResource();
}

void WritableTexture::OnResize(UINT newWidth, UINT newHeight)
{
	if((mWidth != newWidth) || (mHeight != newHeight))
	{
		mWidth = newWidth;
		mHeight = newHeight;

		mViewport = { 0.0f, 0.0f, (float)newWidth, (float)newHeight, 0.0f, 1.0f };
		mScissorRect = { 0, 0, (int)newWidth, (int)newHeight };

		BuildResource();

		// New resource, so we need new descriptors to that resource.
		BuildDescriptors();
	}
}
 
void WritableTexture::BuildResource()
{
	D3D12_RESOURCE_DESC texDesc = {};
	ZeroMemory(&texDesc, sizeof(D3D12_RESOURCE_DESC));

	texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	texDesc.Alignment = 0;
	texDesc.Width = mWidth;
	texDesc.Height = mHeight;
	texDesc.DepthOrArraySize = 1;
	texDesc.MipLevels = 1;
	texDesc.Format = mFormat;
	texDesc.SampleDesc.Count = 1;
	texDesc.SampleDesc.Quality = 0;
	texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	texDesc.Flags = mFlag;

	//const auto texDesc = CD3DX12_RESOURCE_DESC::Tex2D(texFormat, Tex->Width, Tex->Height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
	//ThrowIfFailed(m_d3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_SHARED,
	//	&texDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&Tex->Resource)));

	D3D12_CLEAR_VALUE optClear;
	D3D12_CLEAR_VALUE* optPtr = nullptr;
	optClear.Format = mClearFormat;
	if (textureType == DepthStencil)
	{
		optClear.DepthStencil.Depth = 1.0f;
		optClear.DepthStencil.Stencil = 0;
		optPtr = &optClear;
	}
	else if (textureType == RenderTarget)
	{
		memcpy(optClear.Color, &ClearColor, 4 * sizeof(float));
		optPtr = &optClear;
	}
	else if (textureType == UnorderedAccess)
	{
		optPtr = nullptr;
	}

	ThrowIfFailed(md3dDevice->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
		mHeapFlag,
		&texDesc,
		D3D12_RESOURCE_STATE_COMMON,
		optPtr,
		IID_PPV_ARGS(&mWritableTexture)));
}

void WritableTexture::BuildDescriptors()
{
    // Create SRV to resource so we can sample the shadow map in a shader program.
	CreateSrvDescriptor(mhCpuSrv, mhGpuSrv);
	// Create RTV to resource so we can use it as Render Target
	CreateRtvDescriptor(mhCpuRtv);
	// Create DSV to resource so we can render to the shadow map.
	CreateDsvDescriptor(mhCpuDsv);
	// Create UAV to resource
	CreateUavDescriptor(mhCpuUav, mhGpuUav);
}

void WritableTexture::CreateSrvDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE hCpuSrv, CD3DX12_GPU_DESCRIPTOR_HANDLE hGpuSrv)
{
	mhCpuSrv = hCpuSrv;
	mhGpuSrv = hGpuSrv;

	// Create SRV to resource so we can sample the shadow map in a shader program.
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Format = mSRFormat;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = 1;
	srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
	srvDesc.Texture2D.PlaneSlice = 0;
	md3dDevice->CreateShaderResourceView(mWritableTexture.Get(), &srvDesc, mhCpuSrv);
}

void WritableTexture::CreateRtvDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE hCpuRtv)
{
	if (textureType != RenderTarget)
		return;

	mhCpuRtv = hCpuRtv;

	// Create RTV to resource so we can use it as Render Target
	md3dDevice->CreateRenderTargetView(mWritableTexture.Get(), nullptr, mhCpuRtv);
}

void WritableTexture::CreateDsvDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE hCpuDsv)
{
	if (textureType != DepthStencil)
		return;

	mhCpuDsv = hCpuDsv;

	// Create DSV to resource so we can render to the shadow map.
	D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc;
	dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
	dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	dsvDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	dsvDesc.Texture2D.MipSlice = 0;
	md3dDevice->CreateDepthStencilView(mWritableTexture.Get(), &dsvDesc, mhCpuDsv);
}

void WritableTexture::CreateUavDescriptor(CD3DX12_CPU_DESCRIPTOR_HANDLE hCpuUav, CD3DX12_GPU_DESCRIPTOR_HANDLE hGpuUav)
{
	if (textureType != UnorderedAccess)
		return;

	mhCpuUav = hCpuUav;
	mhGpuUav = hGpuUav;

	// Create DSV to resource so we can render to the shadow map.
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	uavDesc.Format = mFormat;
	uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	uavDesc.Texture2D.MipSlice = 0;
	md3dDevice->CreateUnorderedAccessView(mWritableTexture.Get(), nullptr, &uavDesc, mhCpuUav);
}

void WritableTexture::CaptureTexture(const wchar_t* name, const GUID& format, ID3D12CommandQueue* ptrCmdQueue)
{
	DirectX::SaveWICTextureToFile(
		ptrCmdQueue,
		mWritableTexture.Get(),
		format,
		name
	);
}

void WritableTexture::ChangeResourceState(ID3D12GraphicsCommandList* ptr_CommandList, D3D12_RESOURCE_STATES NextState)
{
	if (CurrState == NextState)
		return;
	ptr_CommandList->ResourceBarrier(1,	//BarrierÆÁÕÏ¸öÊý
		&CD3DX12_RESOURCE_BARRIER::Transition(
			mWritableTexture.Get(),
			CurrState,
			NextState));
	CurrState = NextState;
}

UINT WritableTexture::Width()const
{
	return mWidth;
}

UINT WritableTexture::Height()const
{
	return mHeight;
}

ID3D12Resource* WritableTexture::Resource()
{
	return mWritableTexture.Get();
}

CD3DX12_CPU_DESCRIPTOR_HANDLE WritableTexture::Srv()const
{
	return mhCpuSrv;
}

CD3DX12_CPU_DESCRIPTOR_HANDLE WritableTexture::Rtv()const
{
	return mhCpuRtv;
}

CD3DX12_CPU_DESCRIPTOR_HANDLE WritableTexture::Dsv()const
{
	return mhCpuDsv;
}

CD3DX12_GPU_DESCRIPTOR_HANDLE WritableTexture::gpuSrv() const
{
	return mhGpuSrv;
}

CD3DX12_GPU_DESCRIPTOR_HANDLE WritableTexture::gpuUav() const
{
	return mhGpuUav;
}

D3D12_VIEWPORT WritableTexture::Viewport()const
{
	return mViewport;
}

D3D12_RECT WritableTexture::ScissorRect()const
{
	return mScissorRect;
}