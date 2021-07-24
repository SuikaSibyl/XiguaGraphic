#pragma once
#include "CudaUtil.h"
#include <Windows.h>
#include <Utility.h>
#include <Texture.h>
#include <SynchronizationModule.h>
#include <Camera.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Scene.h>

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
	~CudaManager();
	void MoveToNextFrame(Camera* camera);
	void InitCuda();
	void InitSynchronization(D3DModules::SynchronizationModule* syncModule);

	void SetTexture(WritableTexture* texture);
	void SetTexture(Texture* texture);

	void BindTexture(Texture* texture);
	void BindCubemap(Texture* texture);
	void BindTexture(Texture* texture, Texture* compare);

	void CreateScene(Suika::Scene& scene);
	float* GetEnvCoeff();

	void SetEnvmap();

	Texture* basictexture;
	Texture* textureenv;
	WritableTexture* texture;
private:
	// CUDA objects
	cudaExternalMemoryHandleType m_externalMemoryHandleType;
	cudaExternalMemory_t	     m_externalMemory;
	cudaExternalMemory_t	     m_externalMemoryEnvmap;
	cudaExternalMemory_t	     m_externalMemoryEnvmap2;
	cudaExternalMemory_t	     m_externalMemoryCubemap;
	cudaExternalSemaphore_t      m_externalSemaphore;
	cudaStream_t				 m_streamToRun;
	LUID						 m_dx12deviceluid;
	UINT						 m_cudaDeviceID;
	UINT						 m_nodeMask;
	float						 m_AnimTime;
	int							m_time = 0;
	void* m_cudaDevVertptr = NULL;

	void GetHardwareAdapter(IDXGIFactory2* pFactory, IDXGIAdapter1** ppAdapter);
	void UpdateCudaSurface(Camera* camera = nullptr);
	void InitCudaDatas();

	D3DModules::SynchronizationModule* m_SyncModule;
	ID3D12Device* m_d3dDevice;
	float* dev_triangle_p;
};