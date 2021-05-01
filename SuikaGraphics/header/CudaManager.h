#pragma once
#include "CudaUtil.h"
#include <Windows.h>
#include <Utility.h>
#include <Texture.h>
#include <SynchronizationModule.h>
#include <Camera.h>

class WindowsSecurityAttributes {
protected:
	SECURITY_ATTRIBUTES m_winSecurityAttributes;
	PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
	WindowsSecurityAttributes();
	~WindowsSecurityAttributes();
	SECURITY_ATTRIBUTES* operator&();
};

class CudaManager
{
public:
	CudaManager(IDXGIFactory4* factory, ID3D12Device* device);
	void MoveToNextFrame(Camera* camera);
	void InitCuda();
	void InitSynchronization(D3DModules::SynchronizationModule* syncModule);

	void SetTexture(Texture* texture, const CD3DX12_RESOURCE_DESC* desc);
	Texture* texture;
private:
	// CUDA objects
	cudaExternalMemoryHandleType m_externalMemoryHandleType;
	cudaExternalMemory_t	     m_externalMemory;
	cudaExternalSemaphore_t      m_externalSemaphore;
	cudaStream_t				 m_streamToRun;
	LUID						 m_dx12deviceluid;
	UINT						 m_cudaDeviceID;
	UINT						 m_nodeMask;
	float						 m_AnimTime;
	int							m_time = 0;
	void* m_cudaDevVertptr = NULL;

	void GetHardwareAdapter(IDXGIFactory2* pFactory, IDXGIAdapter1** ppAdapter);
	void UpdateCudaSurface(Texture* texture, Camera* camera = nullptr);

	D3DModules::SynchronizationModule* m_SyncModule;
	ID3D12Device* m_d3dDevice;
};