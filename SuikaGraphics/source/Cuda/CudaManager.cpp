#include <exception>
#include "CudaManager.h"
#include <Windows.h>
#include <Debug.h>
#include <aclapi.h>
#include <CudaFunc.h>

WindowsSecurityAttributes::WindowsSecurityAttributes()
{
	m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void**));
	assert(m_winPSecurityDescriptor != (PSECURITY_DESCRIPTOR)NULL);

	PSID* ppSID = (PSID*)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

	InitializeSecurityDescriptor(m_winPSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);

	SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
	AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, ppSID);

	EXPLICIT_ACCESS explicitAccess;
	ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
	explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
	explicitAccess.grfAccessMode = SET_ACCESS;
	explicitAccess.grfInheritance = INHERIT_ONLY;
	explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
	explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
	explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

	SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

	SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);

	m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
	m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
	m_winSecurityAttributes.bInheritHandle = TRUE;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes()
{
	PSID* ppSID = (PSID*)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

	if (*ppSID) {
		FreeSid(*ppSID);
	}
	if (*ppACL) {
		LocalFree(*ppACL);
	}
	free(m_winPSecurityDescriptor);
}

SECURITY_ATTRIBUTES*
WindowsSecurityAttributes::operator&()
{
	return &m_winSecurityAttributes;
}

CudaManager::CudaManager(IDXGIFactory4* factory, ID3D12Device* device)
{
	bool m_useWarpDevice = false;
	m_d3dDevice = device;

	if (m_useWarpDevice)
	{
		ComPtr<IDXGIAdapter> warpAdapter;
		ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));
	}
	else
	{
		ComPtr<IDXGIAdapter1> hardwareAdapter;
		GetHardwareAdapter(factory, &hardwareAdapter);

		DXGI_ADAPTER_DESC1 desc;
		hardwareAdapter->GetDesc1(&desc);
		m_dx12deviceluid = desc.AdapterLuid;
	}

	InitCuda();
}

void CudaManager::MoveToNextFrame(Camera* camera)
{
	const UINT64 currentFenceValue = m_SyncModule->mCurrentFence;

	//cudaExternalSemaphoreWaitParams externalSemaphoreWaitParams{};
	//externalSemaphoreWaitParams.params.fence.value = currentFenceValue;
	//checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_externalSemaphore, &externalSemaphoreWaitParams, 1, m_streamToRun));

	m_AnimTime += .1;
	UpdateCudaSurface(texture, camera);

	//m_SyncModule->mCurrFrameResource->fenceCPU = ++m_SyncModule->mCurrentFence;

	//cudaExternalSemaphoreSignalParams externalSemaphoreSignalParams{};
	//externalSemaphoreSignalParams.params.fence.value = m_SyncModule->mCurrFrameResource->fenceCPU;
	//checkCudaErrors(cudaSignalExternalSemaphoresAsync(&m_externalSemaphore, &externalSemaphoreSignalParams, 1, m_streamToRun));
}

void CudaManager::InitCuda()
{
	int num_cuda_devices = 0;
	checkCudaErrors(cudaGetDeviceCount(&num_cuda_devices));

	if (!num_cuda_devices)
	{
		throw std::exception("No CUDA Devices found");
	}
	for (UINT devId = 0; devId < num_cuda_devices; devId++)
	{
		cudaDeviceProp devProp;
		checkCudaErrors(cudaGetDeviceProperties(&devProp, devId));

		if ((memcmp(&m_dx12deviceluid.LowPart, devProp.luid, sizeof(m_dx12deviceluid.LowPart)) == 0) && (memcmp(&m_dx12deviceluid.HighPart, devProp.luid + sizeof(m_dx12deviceluid.LowPart), sizeof(m_dx12deviceluid.HighPart)) == 0))
		{
			checkCudaErrors(cudaSetDevice(devId));
			m_cudaDeviceID = devId;
			m_nodeMask = devProp.luidDeviceNodeMask;
			checkCudaErrors(cudaStreamCreate(&m_streamToRun));
			printf("CUDA Device Used [%d] %s\n", devId, devProp.name);
			Debug::LogSystem(QString::fromStdString("CUDA Device Used" + std::to_string(devId) + " " + devProp.name));

			size_t prev_size;
			size_t setting_size = 4096;
			checkCudaErrors(cudaDeviceGetLimit(&prev_size, cudaLimitStackSize));
			Debug::LogSystem(QString::fromStdString("CUDA Limit Stack Size: ") + QString::number(prev_size));
			checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, setting_size));
			Debug::LogSystem(QString::fromStdString("CUDA Alter Stack Size: ") + QString::number(setting_size));
			break;
		}
	}
}

void CudaManager::InitSynchronization(D3DModules::SynchronizationModule* syncModule)
{
	m_SyncModule = syncModule;

	// Create cudaExternalSemaphoreHandleDesc
	cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc{};
	memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
	WindowsSecurityAttributes windowsSecurityAttributes;

	// Create Synchronization
	LPCWSTR name{};
	HANDLE sharedHandle{};
	externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
	m_d3dDevice->CreateSharedHandle((syncModule->m_fence).Get(), &windowsSecurityAttributes, GENERIC_ALL, name, &sharedHandle);
	externalSemaphoreHandleDesc.handle.win32.handle = sharedHandle;
	externalSemaphoreHandleDesc.flags = 0;
	checkCudaErrors(cudaImportExternalSemaphore(&m_externalSemaphore, &externalSemaphoreHandleDesc));
}

void CudaManager::GetHardwareAdapter(IDXGIFactory2* pFactory, IDXGIAdapter1** ppAdapter)
{
	ComPtr<IDXGIAdapter1> adapter;
	*ppAdapter = nullptr;

	for (UINT adapterIndex = 0; DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(adapterIndex, &adapter); ++adapterIndex)
	{
		DXGI_ADAPTER_DESC1 desc;
		adapter->GetDesc1(&desc);

		if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
		{
			// Don't select the Basic Render Driver adapter.
			// If you want a software adapter, pass in "/warp" on the command line.
			continue;
		}

		// Check to see if the adapter supports Direct3D 12, but don't create the
		// actual device yet.
		if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
		{
			break;
		}
	}

	*ppAdapter = adapter.Detach();
}


void CudaManager::SetTexture(Texture* texture, const CD3DX12_RESOURCE_DESC* desc)
{
	//HANDLE sharedHandle;
	//WindowsSecurityAttributes windowsSecurityAttributes;
	//LPCWSTR name = NULL;
	//ThrowIfFailed(m_d3dDevice->CreateSharedHandle(texture->Resource.Get(), &windowsSecurityAttributes, GENERIC_ALL, name, &sharedHandle));

	//D3D12_RESOURCE_ALLOCATION_INFO d3d12ResourceAllocationInfo;
	//d3d12ResourceAllocationInfo = m_d3dDevice->GetResourceAllocationInfo(m_nodeMask, 1, &CD3DX12_RESOURCE_DESC::Buffer(texture->pixelBufferSize));
	//auto actualSize = d3d12ResourceAllocationInfo.SizeInBytes;

	HANDLE sharedHandle{};
	WindowsSecurityAttributes secAttr{};
	ThrowIfFailed(m_d3dDevice->CreateSharedHandle(texture->Resource.Get(), &secAttr, GENERIC_ALL, 0, &sharedHandle));
	const auto texAllocInfo = m_d3dDevice->GetResourceAllocationInfo(m_nodeMask, 1, desc);

	cudaExternalMemoryHandleDesc cuExtmemHandleDesc{};
	cuExtmemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
	cuExtmemHandleDesc.handle.win32.handle = sharedHandle;
	cuExtmemHandleDesc.size = texAllocInfo.SizeInBytes;
	cuExtmemHandleDesc.flags = cudaExternalMemoryDedicated;
	checkCudaErrors(cudaImportExternalMemory(&m_externalMemory, &cuExtmemHandleDesc));

	cudaExternalMemoryMipmappedArrayDesc cuExtmemMipDesc{};
	cuExtmemMipDesc.extent = make_cudaExtent(texture->Width, texture->Height, 0);
	cuExtmemMipDesc.formatDesc = cudaCreateChannelDesc<float4>();
	cuExtmemMipDesc.numLevels = 1;
	cuExtmemMipDesc.flags = cudaArraySurfaceLoadStore;

	cudaMipmappedArray_t cuMipArray{};
	checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(&cuMipArray, m_externalMemory, &cuExtmemMipDesc));

	cudaArray_t cuArray{};
	checkCudaErrors(cudaGetMipmappedArrayLevel(&cuArray, cuMipArray, 0));

	cudaResourceDesc cuResDesc{};
	cuResDesc.resType = cudaResourceTypeArray;
	cuResDesc.res.array.array = cuArray;
	checkCudaErrors(cudaCreateSurfaceObject(&(texture->cuSurface), &cuResDesc));
	// where cudaSurfaceObject_t cuSurface{};

	m_AnimTime = 1.0f;
	this->texture = texture;
	//UpdateCudaSurface(texture);

	//checkCudaErrors(cudaStreamSynchronize(m_streamToRun));
	//cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
	//memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));
	//externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
	//externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
	//externalMemoryHandleDesc.size = actualSize;
	//externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

	//checkCudaErrors(cudaImportExternalMemory(&m_externalMemory, &externalMemoryHandleDesc));

	//cudaExternalMemoryBufferDesc externalMemoryBufferDesc;
	//memset(&externalMemoryBufferDesc, 0, sizeof(externalMemoryBufferDesc));
	//externalMemoryBufferDesc.offset = 0;
	//externalMemoryBufferDesc.size = texture->pixelBufferSize;
	//externalMemoryBufferDesc.flags = 0;

	//checkCudaErrors(cudaExternalMemoryGetMappedBuffer(&m_cudaDevVertptr, m_externalMemory, &externalMemoryBufferDesc));
	////RunKernel(TextureWidth, TextureHeight, (float*)m_cudaDevVertptr, m_streamToRun, 1.0f);
	//checkCudaErrors(cudaStreamSynchronize(m_streamToRun));

	return;
}

extern "C" void RunKernel(size_t textureW, size_t textureH, cudaSurfaceObject_t surfaceObject, cudaStream_t streamToRun, int i, float* paras);

void CudaManager::UpdateCudaSurface(Texture* texture, Camera* camera)
{
	if (camera->camUpdate)
	{
		camera->camUpdate = false;
		m_time = 0;
		RunKernel(texture->Width, texture->Height, texture->cuSurface, m_streamToRun, m_time++, camera->camParas);
	}
	else
	{
		RunKernel(texture->Width, texture->Height, texture->cuSurface, m_streamToRun, m_time++, camera->camParas);
	}
}