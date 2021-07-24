#pragma once
#include <Utility.h>

template<typename T> 
class UploadBuffer
{
public: 
	/// <summary>
	/// Create an Upload Buffer with Type T & size elementCount
	/// </summary>
	/// <param name="elementCount"> Number of upload elements </param>
	UploadBuffer(ID3D12Device* device, UINT elementCount, bool isConstantBuffer) : 
		mIsConstantBuffer(isConstantBuffer) 
	{
		// Size of one element
		mElementByteSize = sizeof(T); 

		// Constant buffer elements need to be multiples of 256 bytes. 
		// This is because the hardware can only view constant data 
		// at m*256 byte offsets and of n*256 byte lengths.
		if(isConstantBuffer) 
			mElementByteSize = Utils::CalcConstantBufferByteSize(sizeof(T));
		
		// Create the Upload Buffer
		ThrowIfFailed(device->CreateCommittedResource(
			// Type: UploadResource
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
			// Define the size
			D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(mElementByteSize*elementCount), 
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&mUploadBuffer))); 

		// obtain a pointer to the resource data
		ThrowIfFailed(mUploadBuffer->Map(0, nullptr, reinterpret_cast<void**>(&mMappedData)));
		// We do not need to unmap until we are done with the resource. 
		// However, we must not write to the resource while it is in use by 
		// the GPU (so we must use synchronization techniques). 
	}

	UploadBuffer(const UploadBuffer& rhs) = delete; 
	UploadBuffer& operator=(const UploadBuffer& rhs) = delete; 

	~UploadBuffer() { 
		// Unmap the CPU pointer
		if(mUploadBuffer != nullptr) 
			mUploadBuffer->Unmap(0, nullptr); mMappedData = nullptr; 
	}

	ID3D12Resource* Resource()const 
	{ 
		return mUploadBuffer.Get(); 
	}

	/// <summary>
	/// Copy one data to the buffer
	/// </summary>
	/// <param name="elementIndex"> Destination offset index </param>
	void CopyData(int elementIndex, const T& data) 
	{ 
		memcpy(&mMappedData[elementIndex*mElementByteSize], &data, sizeof(T)); 
	} 

private: 
	Microsoft::WRL::ComPtr<ID3D12Resource> mUploadBuffer;

	// Mapped data on CPU
	BYTE* mMappedData = nullptr;
	// Size of one element
	UINT mElementByteSize = 0; 
	// Is CONSTANT BUFFER
	bool mIsConstantBuffer = false;
};