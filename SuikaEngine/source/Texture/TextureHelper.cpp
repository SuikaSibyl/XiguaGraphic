#include <Precompiled.h>
#include <TextureHelper.h>
#include <QDirect3D12Widget.h>

using namespace IMG;

#define TEXTURE_PATH L"../Resources/Textures/"

TextureHelper::TextureHelper(QDirect3D12Widget* qd3d)
{
	m_qd3dWidget = qd3d;
}

std::unique_ptr<Texture> TextureHelper::CreateCubemapArray(std::string name, std::wstring filepath)
{
	m_d3dDevice = m_qd3dWidget->m_d3dDevice.Get();
	m_CommandList = m_qd3dWidget->m_CommandList;

	std::unique_ptr<Texture> Tex = std::make_unique<Texture>();
	Tex->Name = name;
	Tex->type = Texture::Type::CubemapArray;

	std::wstring postfix = filepath.substr(filepath.size() - 3);
	std::wstring prename = filepath.substr(0, filepath.size() - 4);

	// If it's a dss texuter
	if (postfix == L"dds" || postfix == L"DDS")
	{
		Tex->Filename = TEXTURE_PATH + filepath;

		ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(
			m_d3dDevice, m_CommandList,
			Tex->Filename.c_str(),
			Tex->Resource, Tex->UploadHeap));
	}
	else
	{
		Debug::LogError("CubemapArrayRead Only support DDS!");
	}

	return std::move(Tex);
}

std::unique_ptr<Texture> TextureHelper::CreateCubemapTexture(std::string name, std::wstring filepath)
{
	m_d3dDevice = m_qd3dWidget->m_d3dDevice.Get();
	m_CommandList = m_qd3dWidget->m_CommandList;

	std::unique_ptr<Texture> Tex = std::make_unique<Texture>();
	Tex->Name = name;
	Tex->type = Texture::Type::Cubemap;

	std::wstring postfix = filepath.substr(filepath.size() - 3);
	std::wstring prename = filepath.substr(0, filepath.size() - 4);

	// If it's a dss texuter
	if (postfix == L"dds" || postfix == L"DDS")
	{
		Tex->Filename = TEXTURE_PATH + filepath;

		ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(
			m_d3dDevice, m_CommandList,
			Tex->Filename.c_str(),
			Tex->Resource, Tex->UploadHeap));
	}
	else
	{
		Tex->Filename = TEXTURE_PATH + prename;
		CubemapImage image = ImageHelper::ReadCubemapPic(Tex->Filename, postfix);
		// Summarize the size of v&i
		const UINT  pixel_data_size = (UINT)image.sub_pixels[0].size() * sizeof(Color4<uint8_t>) * 6;

		//ΪTexture����Resource
		D3D12_RESOURCE_DESC texDesc;
		memset(&texDesc, 0, sizeof(D3D12_RESOURCE_DESC));
		texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		texDesc.Alignment = 0;
		texDesc.Width = (uint32_t)image.header.width;
		texDesc.Height = (uint32_t)image.header.height;
		texDesc.DepthOrArraySize = 6;
		texDesc.MipLevels = 1;
		texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		texDesc.SampleDesc.Count = 1;
		texDesc.SampleDesc.Quality = 0;
		texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

		UINT num2DSubresources = texDesc.DepthOrArraySize * texDesc.MipLevels;

		//Ĭ�϶�
		// ����Ĭ�϶ѡ�������Դ�Ķѷֺܶ������ͣ�Ĭ�϶ѣ��ϴ��ѣ�Ĭ�϶��ϵ�������ԴȨ����GPUAvailable��CPUUnAvailable�ġ�
		D3D12_HEAP_PROPERTIES heap;
		memset(&heap, 0, sizeof(heap));
		heap.Type = D3D12_HEAP_TYPE_DEFAULT;

		//���ﴴ����ʱ���ָ����COPY_DEST״̬�����������Ҫ����Դ���ϰ�������Ū��ֻ��
		ThrowIfFailed(m_d3dDevice->CreateCommittedResource(
			&heap,
			D3D12_HEAP_FLAG_NONE,
			&texDesc,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(&Tex->Resource)
		));

		//��ȡfootprint
		D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint[6];
		UINT64  total_bytes = 0;
		m_d3dDevice->GetCopyableFootprints(&texDesc, 0, texDesc.DepthOrArraySize, 0, footprint, nullptr, nullptr, &total_bytes);

		//ΪUploadTexture������Դ
		D3D12_RESOURCE_DESC uploadTexDesc;
		memset(&uploadTexDesc, 0, sizeof(uploadTexDesc));
		uploadTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		uploadTexDesc.Width = total_bytes * 6;
		uploadTexDesc.Height = 1;
		uploadTexDesc.DepthOrArraySize = 1;
		uploadTexDesc.MipLevels = 1;
		uploadTexDesc.SampleDesc.Count = 1;
		uploadTexDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

		//��7�������ϴ���
		D3D12_HEAP_PROPERTIES  uploadheap;
		memset(&uploadheap, 0, sizeof(uploadheap));
		uploadheap.Type = D3D12_HEAP_TYPE_UPLOAD;

		//��8�����ϴ����ϴ�����Դ
		ThrowIfFailed(m_d3dDevice->CreateCommittedResource(
			&uploadheap,
			D3D12_HEAP_FLAG_NONE,
			&uploadTexDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&Tex->UploadHeap)
		));

		//m_CommandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);

		D3D12_SUBRESOURCE_DATA subResourceDatas = {};
		subResourceDatas.RowPitch = image.header.width * sizeof(Color4<uint8_t>);
		subResourceDatas.SlicePitch = pixel_data_size / 6;
		////���ĺ���UpdateSubresources�������ݴ�CPU�ڴ濽�����ϴ��ѣ��ٴ��ϴ��ѿ�����Ĭ�϶ѡ�1����������Դ���±꣨ģ���ж��壬��Ϊ��2������Դ��
		//UpdateSubresources<6>(m_CommandList, Tex->Resource.Get(), Tex->UploadHeap.Get(), 0, 0, num2DSubresources, &subResourceDatas);
		// Use Heap-allocating UpdateSubresources implementation for variable number of subresources (which is the case for textures).
		for (int i = 0; i < 6; i++)
		{
			subResourceDatas.pData = image.sub_pixels[i].data();
			UpdateSubresources(m_CommandList, Tex->Resource.Get(), Tex->UploadHeap.Get(), footprint[i].Offset, i, 1, &subResourceDatas);
		}

		//������Դ����
		D3D12_RESOURCE_BARRIER barrier;
		memset(&barrier, 0, sizeof(barrier));
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier.Transition.pResource = Tex->Resource.Get();
		barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
		m_CommandList->ResourceBarrier(1, &barrier);
	}

	return std::move(Tex);
}

std::unique_ptr<Texture> TextureHelper::CreateTexture(std::string name, std::wstring filepath)
{
	m_d3dDevice = m_qd3dWidget->m_d3dDevice.Get();
	m_CommandList = m_qd3dWidget->m_CommandList;

	std::unique_ptr<Texture> Tex = std::make_unique<Texture>();
	Tex->Name = name;
	Tex->type = Texture::Type::Texture2D;

	std::wstring postfix = filepath.substr(filepath.size() - 3);

	// If it's a dss texuter
	if (postfix == L"dds" || postfix == L"DDS")
	{
		Tex->Filename = TEXTURE_PATH + filepath;

		ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(
			m_d3dDevice, m_CommandList,
			Tex->Filename.c_str(),
			Tex->Resource, Tex->UploadHeap));
	}
	else if (postfix == L"hdr")
	{
		Tex->Filename = TEXTURE_PATH + filepath;
		HDRImage image = ImageHelper::ReadHDRPic(Tex->Filename);
		// Summarize the size of v&i
		const UINT  pixel_data_size = (UINT)image.pixels.size() * sizeof(Color4<float>);

		//ΪTexture����Resource
		D3D12_RESOURCE_DESC texDesc;
		memset(&texDesc, 0, sizeof(D3D12_RESOURCE_DESC));
		texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		texDesc.Alignment = 0;
		texDesc.Width = (uint32_t)image.header.width;
		texDesc.Height = (uint32_t)image.header.height;
		texDesc.DepthOrArraySize = 1;
		texDesc.MipLevels = 1;
		texDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		texDesc.SampleDesc.Count = 1;
		texDesc.SampleDesc.Quality = 0;
		texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

		Tex->Width = texDesc.Width;
		Tex->Height = texDesc.Height;
		Tex->Channels = 4;
		Tex->bindCuda = true;

		//Ĭ�϶�
		// ����Ĭ�϶ѡ�������Դ�Ķѷֺܶ������ͣ�Ĭ�϶ѣ��ϴ��ѣ�Ĭ�϶��ϵ�������ԴȨ����GPUAvailable��CPUUnAvailable�ġ�
		D3D12_HEAP_PROPERTIES heap;
		memset(&heap, 0, sizeof(heap));
		heap.Type = D3D12_HEAP_TYPE_DEFAULT;

		//���ﴴ����ʱ���ָ����COPY_DEST״̬�����������Ҫ����Դ���ϰ�������Ū��ֻ��
		ThrowIfFailed(m_d3dDevice->CreateCommittedResource(
			&heap,
			D3D12_HEAP_FLAG_SHARED,
			&texDesc,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(&Tex->Resource)
		));

		//��ȡfootprint
		D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
		UINT64  total_bytes = 0;
		m_d3dDevice->GetCopyableFootprints(&texDesc, 0, 1, 0, &footprint, nullptr, nullptr, &total_bytes);

		//ΪUploadTexture������Դ
		D3D12_RESOURCE_DESC uploadTexDesc;
		memset(&uploadTexDesc, 0, sizeof(uploadTexDesc));
		uploadTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		uploadTexDesc.Width = total_bytes;
		uploadTexDesc.Height = 1;
		uploadTexDesc.DepthOrArraySize = 1;
		uploadTexDesc.MipLevels = 1;
		uploadTexDesc.SampleDesc.Count = 1;
		uploadTexDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

		//��7�������ϴ���
		D3D12_HEAP_PROPERTIES  uploadheap;
		memset(&uploadheap, 0, sizeof(uploadheap));
		uploadheap.Type = D3D12_HEAP_TYPE_UPLOAD;

		//��8�����ϴ����ϴ�����Դ
		ThrowIfFailed(m_d3dDevice->CreateCommittedResource(
			&uploadheap,
			D3D12_HEAP_FLAG_NONE,
			&uploadTexDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&Tex->UploadHeap)
		));

		//m_CommandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);

		D3D12_SUBRESOURCE_DATA subResourceData = {};
		subResourceData.pData = image.pixels.data();
		subResourceData.RowPitch = image.header.width * sizeof(Color4<float>);
		subResourceData.SlicePitch = pixel_data_size;

		//���ĺ���UpdateSubresources�������ݴ�CPU�ڴ濽�����ϴ��ѣ��ٴ��ϴ��ѿ�����Ĭ�϶ѡ�1����������Դ���±꣨ģ���ж��壬��Ϊ��2������Դ��
		UpdateSubresources<1>(m_CommandList, Tex->Resource.Get(), Tex->UploadHeap.Get(), 0, 0, 1, &subResourceData);

		//������Դ����
		D3D12_RESOURCE_BARRIER barrier;
		memset(&barrier, 0, sizeof(barrier));
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier.Transition.pResource = Tex->Resource.Get();
		barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
		m_CommandList->ResourceBarrier(1, &barrier);
	}
	else
	{
		Tex->Filename = TEXTURE_PATH + filepath;

		Image image = ImageHelper::ReadPic(Tex->Filename, postfix);
		// Summarize the size of v&i
		const UINT  pixel_data_size = (UINT)image.pixels.size() * sizeof(Color4<uint8_t>);

		//ΪTexture����Resource
		D3D12_RESOURCE_DESC texDesc;
		memset(&texDesc, 0, sizeof(D3D12_RESOURCE_DESC));
		texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		texDesc.Alignment = 0;
		texDesc.Width = (uint32_t)image.header.width;
		texDesc.Height = (uint32_t)image.header.height;

		Tex->Width = texDesc.Width;
		Tex->Height = texDesc.Height;
		Tex->Channels = 4;
		Tex->bindCuda = true;

		texDesc.DepthOrArraySize = 1;
		texDesc.MipLevels = 1;
		texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		texDesc.SampleDesc.Count = 1;
		texDesc.SampleDesc.Quality = 0;
		texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

		//Ĭ�϶�
		// ����Ĭ�϶ѡ�������Դ�Ķѷֺܶ������ͣ�Ĭ�϶ѣ��ϴ��ѣ�Ĭ�϶��ϵ�������ԴȨ����GPUAvailable��CPUUnAvailable�ġ�
		D3D12_HEAP_PROPERTIES heap;
		memset(&heap, 0, sizeof(heap));
		heap.Type = D3D12_HEAP_TYPE_DEFAULT;

		//���ﴴ����ʱ���ָ����COPY_DEST״̬�����������Ҫ����Դ���ϰ�������Ū��ֻ��
		ThrowIfFailed(m_d3dDevice->CreateCommittedResource(
			&heap,
			D3D12_HEAP_FLAG_SHARED,
			&texDesc,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(&Tex->Resource)
		));

		//��ȡfootprint
		D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
		UINT64  total_bytes = 0;
		m_d3dDevice->GetCopyableFootprints(&texDesc, 0, 1, 0, &footprint, nullptr, nullptr, &total_bytes);

		//ΪUploadTexture������Դ
		D3D12_RESOURCE_DESC uploadTexDesc;
		memset(&uploadTexDesc, 0, sizeof(uploadTexDesc));
		uploadTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		uploadTexDesc.Width = total_bytes;
		uploadTexDesc.Height = 1;
		uploadTexDesc.DepthOrArraySize = 1;
		uploadTexDesc.MipLevels = 1;
		uploadTexDesc.SampleDesc.Count = 1;
		uploadTexDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

		//��7�������ϴ���
		D3D12_HEAP_PROPERTIES  uploadheap;
		memset(&uploadheap, 0, sizeof(uploadheap));
		uploadheap.Type = D3D12_HEAP_TYPE_UPLOAD;

		//��8�����ϴ����ϴ�����Դ
		ThrowIfFailed(m_d3dDevice->CreateCommittedResource(
			&uploadheap,
			D3D12_HEAP_FLAG_NONE,
			&uploadTexDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&Tex->UploadHeap)
		));

		//m_CommandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);

		D3D12_SUBRESOURCE_DATA subResourceData = {};
		subResourceData.pData = image.pixels.data();
		subResourceData.RowPitch = image.header.width * sizeof(Color4<uint8_t>);
		subResourceData.SlicePitch = pixel_data_size;

		//���ĺ���UpdateSubresources�������ݴ�CPU�ڴ濽�����ϴ��ѣ��ٴ��ϴ��ѿ�����Ĭ�϶ѡ�1����������Դ���±꣨ģ���ж��壬��Ϊ��2������Դ��
		UpdateSubresources<1>(m_CommandList, Tex->Resource.Get(), Tex->UploadHeap.Get(), 0, 0, 1, &subResourceData);

		//������Դ����
		D3D12_RESOURCE_BARRIER barrier;
		memset(&barrier, 0, sizeof(barrier));
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier.Transition.pResource = Tex->Resource.Get();
		barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
		m_CommandList->ResourceBarrier(1, &barrier);
	}

	return std::move(Tex);
}

std::unique_ptr<Texture> TextureHelper::CreateCudaTexture(std::string name, UINT m_width, UINT m_height)
{
	std::unique_ptr<Texture> Tex = std::make_unique<Texture>();
	Tex->Channels = 4;
	Tex->Width = m_width;
	Tex->Height = m_height;

	const auto textureSurface = Tex->Width * Tex->Height;
	const auto texturePixels = textureSurface * Tex->Channels;
	const auto textureSizeBytes = sizeof(float) * texturePixels;

	const auto texFormat = Tex->Channels == 4 ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R32G32B32_FLOAT;
	const auto texDesc = CD3DX12_RESOURCE_DESC::Tex2D(texFormat, Tex->Width, Tex->Height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
	ThrowIfFailed(m_d3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_SHARED,
		&texDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&Tex->Resource)));

	Tex->bindCuda = true;

	m_qd3dWidget->m_CudaManagerModule->SetTexture(Tex.get());
	return std::move(Tex);
}

std::array<CD3DX12_STATIC_SAMPLER_DESC, 7> TextureHelper::GetStaticSamplers()
{
	//������POINT,ѰַģʽWRAP�ľ�̬������
	CD3DX12_STATIC_SAMPLER_DESC pointWarp(0,	//��ɫ���Ĵ���
		D3D12_FILTER_MIN_MAG_MIP_POINT,		//����������ΪPOINT(������ֵ)
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//U�����ϵ�ѰַģʽΪWRAP���ظ�Ѱַģʽ��
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//V�����ϵ�ѰַģʽΪWRAP���ظ�Ѱַģʽ��
		D3D12_TEXTURE_ADDRESS_MODE_WRAP);	//W�����ϵ�ѰַģʽΪWRAP���ظ�Ѱַģʽ��

	//������POINT,ѰַģʽCLAMP�ľ�̬������
	CD3DX12_STATIC_SAMPLER_DESC pointClamp(1,	//��ɫ���Ĵ���
		D3D12_FILTER_MIN_MAG_MIP_POINT,		//����������ΪPOINT(������ֵ)
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//U�����ϵ�ѰַģʽΪCLAMP��ǯλѰַģʽ��
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//V�����ϵ�ѰַģʽΪCLAMP��ǯλѰַģʽ��
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP);	//W�����ϵ�ѰַģʽΪCLAMP��ǯλѰַģʽ��

	//������LINEAR,ѰַģʽWRAP�ľ�̬������
	CD3DX12_STATIC_SAMPLER_DESC linearWarp(2,	//��ɫ���Ĵ���
		D3D12_FILTER_MIN_MAG_MIP_LINEAR,		//����������ΪLINEAR(���Բ�ֵ)
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//U�����ϵ�ѰַģʽΪWRAP���ظ�Ѱַģʽ��
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//V�����ϵ�ѰַģʽΪWRAP���ظ�Ѱַģʽ��
		D3D12_TEXTURE_ADDRESS_MODE_WRAP);	//W�����ϵ�ѰַģʽΪWRAP���ظ�Ѱַģʽ��

	//������LINEAR,ѰַģʽCLAMP�ľ�̬������
	CD3DX12_STATIC_SAMPLER_DESC linearClamp(3,	//��ɫ���Ĵ���
		D3D12_FILTER_MIN_MAG_MIP_LINEAR,		//����������ΪLINEAR(���Բ�ֵ)
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//U�����ϵ�ѰַģʽΪCLAMP��ǯλѰַģʽ��
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//V�����ϵ�ѰַģʽΪCLAMP��ǯλѰַģʽ��
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP);	//W�����ϵ�ѰַģʽΪCLAMP��ǯλѰַģʽ��

	//������ANISOTROPIC,ѰַģʽWRAP�ľ�̬������
	CD3DX12_STATIC_SAMPLER_DESC anisotropicWarp(4,	//��ɫ���Ĵ���
		D3D12_FILTER_ANISOTROPIC,			//����������ΪANISOTROPIC(��������)
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//U�����ϵ�ѰַģʽΪWRAP���ظ�Ѱַģʽ��
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//V�����ϵ�ѰַģʽΪWRAP���ظ�Ѱַģʽ��
		D3D12_TEXTURE_ADDRESS_MODE_WRAP);	//W�����ϵ�ѰַģʽΪWRAP���ظ�Ѱַģʽ��

	//������LINEAR,ѰַģʽCLAMP�ľ�̬������
	CD3DX12_STATIC_SAMPLER_DESC anisotropicClamp(5,	//��ɫ���Ĵ���
		D3D12_FILTER_ANISOTROPIC,			//����������ΪANISOTROPIC(��������)
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//U�����ϵ�ѰַģʽΪCLAMP��ǯλѰַģʽ��
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//V�����ϵ�ѰַģʽΪCLAMP��ǯλѰַģʽ��
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP);	//W�����ϵ�ѰַģʽΪCLAMP��ǯλѰַģʽ��

	CD3DX12_STATIC_SAMPLER_DESC shadow(6, // shaderRegister
		D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT, // filter
		D3D12_TEXTURE_ADDRESS_MODE_BORDER,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_BORDER,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_BORDER,  // addressW
		0.0f,                               // mipLODBias
		16,                                 // maxAnisotropy
		D3D12_COMPARISON_FUNC_LESS_EQUAL,
		D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK);

	return{ pointWarp, pointClamp, linearWarp, linearClamp, anisotropicWarp, anisotropicClamp, shadow };
}