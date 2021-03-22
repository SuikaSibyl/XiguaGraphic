#include <TextureHelper.h>
#include <QDirect3D12Widget.h>

using namespace IMG;

TextureHelper::TextureHelper(QDirect3D12Widget* qd3d)
{
	m_qd3dWidget = qd3d;
}

std::unique_ptr<Texture> TextureHelper::CreateTexture(std::string name, std::wstring filepath)
{
	m_d3dDevice = m_qd3dWidget->m_d3dDevice.Get();
	m_CommandList = m_qd3dWidget->m_CommandList.Get();

	std::unique_ptr<Texture> Tex = std::make_unique<Texture>();
	Tex->Name = name;

	std::wstring postfix = filepath.substr(filepath.size() - 3);

	// If it's a dss texuter
	if (postfix == L"dds" || postfix == L"DDS")
	{
		Tex->Filename = L"./Resource/Textures/" + filepath;

		ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(
			m_d3dDevice, m_CommandList,
			Tex->Filename.c_str(),
			Tex->Resource, Tex->UploadHeap));
	}
	else if (postfix == L"hdr")
	{
		Tex->Filename = L"./Resource/Textures/" + filepath;
		HDRImage image = ImageHelper::ReadHDRPic(Tex->Filename);
		// Summarize the size of v&i
		const UINT  pixel_data_size = (UINT)image.pixels.size() * sizeof(Color4<float>);

		//为Texture生成Resource
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
		texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

		//默认堆
		// 创建默认堆。纹理资源的堆分很多种类型，默认堆，上传堆，默认堆上的纹理资源权限是GPUAvailable，CPUUnAvailable的。
		D3D12_HEAP_PROPERTIES heap;
		memset(&heap, 0, sizeof(heap));
		heap.Type = D3D12_HEAP_TYPE_DEFAULT;

		//这里创建的时候就指认了COPY_DEST状态，所以在最后要用资源屏障把它重新弄成只读
		ThrowIfFailed(m_d3dDevice->CreateCommittedResource(
			&heap,
			D3D12_HEAP_FLAG_NONE,
			&texDesc,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(&Tex->Resource)
		));

		//获取footprint
		D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
		UINT64  total_bytes = 0;
		m_d3dDevice->GetCopyableFootprints(&texDesc, 0, 1, 0, &footprint, nullptr, nullptr, &total_bytes);

		//为UploadTexture创建资源
		D3D12_RESOURCE_DESC uploadTexDesc;
		memset(&uploadTexDesc, 0, sizeof(uploadTexDesc));
		uploadTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		uploadTexDesc.Width = total_bytes;
		uploadTexDesc.Height = 1;
		uploadTexDesc.DepthOrArraySize = 1;
		uploadTexDesc.MipLevels = 1;
		uploadTexDesc.SampleDesc.Count = 1;
		uploadTexDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

		//（7）创建上传堆
		D3D12_HEAP_PROPERTIES  uploadheap;
		memset(&uploadheap, 0, sizeof(uploadheap));
		uploadheap.Type = D3D12_HEAP_TYPE_UPLOAD;

		//（8）在上传堆上创建资源
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

		//核心函数UpdateSubresources，将数据从CPU内存拷贝至上传堆，再从上传堆拷贝至默认堆。1是最大的子资源的下标（模板中定义，意为有2个子资源）
		UpdateSubresources<1>(m_CommandList, Tex->Resource.Get(), Tex->UploadHeap.Get(), 0, 0, 1, &subResourceData);

		//插入资源屏障
		D3D12_RESOURCE_BARRIER barrier;
		memset(&barrier, 0, sizeof(barrier));
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier.Transition.pResource = Tex->Resource.Get();
		barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
		m_CommandList->ResourceBarrier(1, &barrier);
	}
	else if (postfix == L"bmp")
	{
		Tex->Filename = L"./Resource/Textures/" + filepath;
		Image image = ImageHelper::ReadPic(Tex->Filename);
		// Summarize the size of v&i
		const UINT  pixel_data_size = (UINT)image.pixels.size() * sizeof(Color4<uint8_t>);

		//为Texture生成Resource
		D3D12_RESOURCE_DESC texDesc;
		memset(&texDesc, 0, sizeof(D3D12_RESOURCE_DESC));
		texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		texDesc.Alignment = 0;
		texDesc.Width = (uint32_t)image.header.width;
		texDesc.Height = (uint32_t)image.header.height;
		texDesc.DepthOrArraySize = 1;
		texDesc.MipLevels = 1;
		texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		texDesc.SampleDesc.Count = 1;
		texDesc.SampleDesc.Quality = 0;
		texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

		//默认堆
		// 创建默认堆。纹理资源的堆分很多种类型，默认堆，上传堆，默认堆上的纹理资源权限是GPUAvailable，CPUUnAvailable的。
		D3D12_HEAP_PROPERTIES heap;
		memset(&heap, 0, sizeof(heap));
		heap.Type = D3D12_HEAP_TYPE_DEFAULT;

		//这里创建的时候就指认了COPY_DEST状态，所以在最后要用资源屏障把它重新弄成只读
		ThrowIfFailed(m_d3dDevice->CreateCommittedResource(
			&heap,
			D3D12_HEAP_FLAG_NONE,
			&texDesc,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(&Tex->Resource)
		));

		//获取footprint
		D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
		UINT64  total_bytes = 0;
		m_d3dDevice->GetCopyableFootprints(&texDesc, 0, 1, 0, &footprint, nullptr, nullptr, &total_bytes);

		//为UploadTexture创建资源
		D3D12_RESOURCE_DESC uploadTexDesc;
		memset(&uploadTexDesc, 0, sizeof(uploadTexDesc));
		uploadTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		uploadTexDesc.Width = total_bytes;
		uploadTexDesc.Height = 1;
		uploadTexDesc.DepthOrArraySize = 1;
		uploadTexDesc.MipLevels = 1;
		uploadTexDesc.SampleDesc.Count = 1;
		uploadTexDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

		//（7）创建上传堆
		D3D12_HEAP_PROPERTIES  uploadheap;
		memset(&uploadheap, 0, sizeof(uploadheap));
		uploadheap.Type = D3D12_HEAP_TYPE_UPLOAD;

		//（8）在上传堆上创建资源
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

		//核心函数UpdateSubresources，将数据从CPU内存拷贝至上传堆，再从上传堆拷贝至默认堆。1是最大的子资源的下标（模板中定义，意为有2个子资源）
		UpdateSubresources<1>(m_CommandList, Tex->Resource.Get(), Tex->UploadHeap.Get(), 0, 0, 1, &subResourceData);

		//插入资源屏障
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

std::array<CD3DX12_STATIC_SAMPLER_DESC, 6> TextureHelper::GetStaticSamplers()
{
	//过滤器POINT,寻址模式WRAP的静态采样器
	CD3DX12_STATIC_SAMPLER_DESC pointWarp(0,	//着色器寄存器
		D3D12_FILTER_MIN_MAG_MIP_POINT,		//过滤器类型为POINT(常量插值)
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//U方向上的寻址模式为WRAP（重复寻址模式）
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//V方向上的寻址模式为WRAP（重复寻址模式）
		D3D12_TEXTURE_ADDRESS_MODE_WRAP);	//W方向上的寻址模式为WRAP（重复寻址模式）

	//过滤器POINT,寻址模式CLAMP的静态采样器
	CD3DX12_STATIC_SAMPLER_DESC pointClamp(1,	//着色器寄存器
		D3D12_FILTER_MIN_MAG_MIP_POINT,		//过滤器类型为POINT(常量插值)
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//U方向上的寻址模式为CLAMP（钳位寻址模式）
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//V方向上的寻址模式为CLAMP（钳位寻址模式）
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP);	//W方向上的寻址模式为CLAMP（钳位寻址模式）

	//过滤器LINEAR,寻址模式WRAP的静态采样器
	CD3DX12_STATIC_SAMPLER_DESC linearWarp(2,	//着色器寄存器
		D3D12_FILTER_MIN_MAG_MIP_LINEAR,		//过滤器类型为LINEAR(线性插值)
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//U方向上的寻址模式为WRAP（重复寻址模式）
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//V方向上的寻址模式为WRAP（重复寻址模式）
		D3D12_TEXTURE_ADDRESS_MODE_WRAP);	//W方向上的寻址模式为WRAP（重复寻址模式）

	//过滤器LINEAR,寻址模式CLAMP的静态采样器
	CD3DX12_STATIC_SAMPLER_DESC linearClamp(3,	//着色器寄存器
		D3D12_FILTER_MIN_MAG_MIP_LINEAR,		//过滤器类型为LINEAR(线性插值)
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//U方向上的寻址模式为CLAMP（钳位寻址模式）
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//V方向上的寻址模式为CLAMP（钳位寻址模式）
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP);	//W方向上的寻址模式为CLAMP（钳位寻址模式）

	//过滤器ANISOTROPIC,寻址模式WRAP的静态采样器
	CD3DX12_STATIC_SAMPLER_DESC anisotropicWarp(4,	//着色器寄存器
		D3D12_FILTER_ANISOTROPIC,			//过滤器类型为ANISOTROPIC(各向异性)
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//U方向上的寻址模式为WRAP（重复寻址模式）
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//V方向上的寻址模式为WRAP（重复寻址模式）
		D3D12_TEXTURE_ADDRESS_MODE_WRAP);	//W方向上的寻址模式为WRAP（重复寻址模式）

	//过滤器LINEAR,寻址模式CLAMP的静态采样器
	CD3DX12_STATIC_SAMPLER_DESC anisotropicClamp(5,	//着色器寄存器
		D3D12_FILTER_ANISOTROPIC,			//过滤器类型为ANISOTROPIC(各向异性)
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//U方向上的寻址模式为CLAMP（钳位寻址模式）
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//V方向上的寻址模式为CLAMP（钳位寻址模式）
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP);	//W方向上的寻址模式为CLAMP（钳位寻址模式）

	IMG::ImageHelper::CreatePic();

	return{ pointWarp, pointClamp, linearWarp, linearClamp, anisotropicWarp, anisotropicClamp };
}