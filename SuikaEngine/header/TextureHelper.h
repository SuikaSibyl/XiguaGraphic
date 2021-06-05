#pragma once

#include <Texture.h>
#include <array>
#include <memory>
#include <Utility.h>
#include <ImageBasic.h>
#include <DDSTextureLoader.h>
#include <string>

class QDirect3D12Widget;

class TextureHelper
{
public:
	static std::array<CD3DX12_STATIC_SAMPLER_DESC, 7> GetStaticSamplers();

	TextureHelper(QDirect3D12Widget* qd3d);

	std::unique_ptr<Texture> CreateCubemapArray(std::string name, std::wstring filepath);
	std::unique_ptr<Texture> CreateCubemapTexture(std::string name, std::wstring filepath);
	std::unique_ptr<Texture> CreateTexture(std::string name, std::wstring filepath);
	std::unique_ptr<Texture> CreateCudaTexture(std::string name, UINT m_width, UINT m_height);

private:
	QDirect3D12Widget* m_qd3dWidget;
	ID3D12Device* m_d3dDevice;
	ID3D12GraphicsCommandList*   m_CommandList;
};