#include <Precompiled.h>
#include "CudaManager.h"
#include <CudaPathTracer.h>

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
	UpdateCudaSurface(camera);

	//m_SyncModule->mCurrFrameResource->fenceCPU = ++m_SyncModule->mCurrentFence;

	//cudaExternalSemaphoreSignalParams externalSemaphoreSignalParams{};
	//externalSemaphoreSignalParams.params.fence.value = m_SyncModule->mCurrFrameResource->fenceCPU;
	//checkCudaErrors(cudaSignalExternalSemaphoresAsync(&m_externalSemaphore, &externalSemaphoreSignalParams, 1, m_streamToRun));
}

void CudaManager::CreateScene(Suika::Scene& scene)
{
	HostCreateScene(scene);
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
	InitCudaDatas();
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


void CudaManager::SetTexture(WritableTexture* texture)
{
	HANDLE sharedHandle{};
	WindowsSecurityAttributes secAttr{};
	ThrowIfFailed(m_d3dDevice->CreateSharedHandle(texture->Resource(), &secAttr, GENERIC_ALL, 0, &sharedHandle));
	const auto texAllocInfo = m_d3dDevice->GetResourceAllocationInfo(m_nodeMask, 1, &texture->Resource()->GetDesc());

	cudaExternalMemoryHandleDesc cuExtmemHandleDesc{};
	cuExtmemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
	cuExtmemHandleDesc.handle.win32.handle = sharedHandle;
	cuExtmemHandleDesc.size = texAllocInfo.SizeInBytes;
	cuExtmemHandleDesc.flags = cudaExternalMemoryDedicated;
	checkCudaErrors(cudaImportExternalMemory(&m_externalMemory, &cuExtmemHandleDesc));

	cudaExternalMemoryMipmappedArrayDesc cuExtmemMipDesc{};
	cuExtmemMipDesc.extent = make_cudaExtent(texture->Width(), texture->Height(), 0);
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

	return;
}


void CudaManager::SetTexture(Texture* texture)
{
	HANDLE sharedHandle{};
	WindowsSecurityAttributes secAttr{};
	ThrowIfFailed(m_d3dDevice->CreateSharedHandle(texture->Resource.Get(), &secAttr, GENERIC_ALL, 0, &sharedHandle));
	const auto texAllocInfo = m_d3dDevice->GetResourceAllocationInfo(m_nodeMask, 1, &texture->Resource.Get()->GetDesc());

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
	this->basictexture = texture;
	//UpdateCudaSurface(texture);

	return;
}

void CudaManager::UpdateCudaSurface(Camera* camera)
{
	if (camera->camUpdate)
	{
		camera->camUpdate = false;
		m_time = 0;
		RunKernel(basictexture->Width, basictexture->Height, basictexture->cuSurface, textureenv->cuSurface, m_streamToRun, m_time++, camera->camParas);
		Debug::LogSystem(QString::number(camera->camParas[0]) + "," + QString::number(camera->camParas[1]) + "," + QString::number(camera->camParas[2]));
	}
	else
	{
		RunKernel(basictexture->Width, basictexture->Height, basictexture->cuSurface, textureenv->cuSurface, m_streamToRun, m_time++, camera->camParas);
	}
}

extern cudaTextureObject_t cubemap_tex;

void CudaManager::BindTexture(Texture* texture)
{
	HANDLE sharedHandle{};
	WindowsSecurityAttributes secAttr{};
	ThrowIfFailed(m_d3dDevice->CreateSharedHandle(texture->Resource.Get(), &secAttr, GENERIC_ALL, 0, &sharedHandle));
	const auto texAllocInfo = m_d3dDevice->GetResourceAllocationInfo(m_nodeMask, 1, &texture->Resource.Get()->GetDesc());

	cudaExternalMemoryHandleDesc cuExtmemHandleDesc{};
	cuExtmemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
	cuExtmemHandleDesc.handle.win32.handle = sharedHandle;
	cuExtmemHandleDesc.size = texAllocInfo.SizeInBytes;
	cuExtmemHandleDesc.flags = cudaExternalMemoryDedicated;
	checkCudaErrors(cudaImportExternalMemory(&m_externalMemoryEnvmap, &cuExtmemHandleDesc));

	cudaExternalMemoryMipmappedArrayDesc cuExtmemMipDesc{};
	cuExtmemMipDesc.extent = make_cudaExtent(texture->Width, texture->Height, 0);
	cuExtmemMipDesc.formatDesc = cudaCreateChannelDesc<float4>();
	cuExtmemMipDesc.numLevels = 1;
	cuExtmemMipDesc.flags = cudaArraySurfaceLoadStore;

	cudaMipmappedArray_t cuMipArray{};
	checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(&cuMipArray, m_externalMemoryEnvmap, &cuExtmemMipDesc));

	cudaArray_t cuArray{};
	checkCudaErrors(cudaGetMipmappedArrayLevel(&cuArray, cuMipArray, 0));

	cudaResourceDesc cuResDesc{};
	cuResDesc.resType = cudaResourceTypeArray;
	cuResDesc.res.array.array = cuArray;
	checkCudaErrors(cudaCreateSurfaceObject(&(texture->cuSurface), &cuResDesc));
	// where cudaSurfaceObject_t cuSurface{};

	this->textureenv = texture;
}

bool LoadFromFile(std::string name, float* lightCoeff)
{
	std::ifstream ifs(name, std::ios::binary | std::ios::in);
	if (!ifs)
	{
		return false;
	}
	ifs.read((char*)lightCoeff, 16 * 3 * sizeof(float));
	ifs.close();
	return true;
}

void WriteToFile(std::string name, float* data)
{
	std::ofstream  ofs(name, std::ios::binary | std::ios::out);
	ofs.write((const char*)data, 16 * 3 * sizeof(float));
	ofs.close();
}

float* CudaManager::GetEnvCoeff()
{
	float* lightCoeff = new float[16 * 3];
	if (!LoadFromFile(TRANSFER_PATH "LightCoeff.transfer", lightCoeff))
	{
		RunPrecomputeEnvironment(lightCoeff, this->textureenv->cuSurface);
		WriteToFile(TRANSFER_PATH "LightCoeff.transfer", lightCoeff);
	}
	for (int i = 0; i < 3; i++)
	{

	}
	return lightCoeff;
}

void CudaManager::BindCubemap(Texture* texture)
{
	// generate input data for layered texture
	unsigned int width = 64, num_faces = 6, num_layers = 1;
	unsigned int cubemap_size = width * width * num_faces;
	unsigned int size = cubemap_size * num_layers * sizeof(float);
	float* h_data = (float*)malloc(size);

	for (int i = 0; i < (int)(cubemap_size * num_layers); i++)
	{
		h_data[i] = (float)i;
	}

	HANDLE sharedHandle{};
	WindowsSecurityAttributes secAttr{};
	ThrowIfFailed(m_d3dDevice->CreateSharedHandle(texture->Resource.Get(), &secAttr, GENERIC_READ, 0, &sharedHandle));
	const auto texAllocInfo = m_d3dDevice->GetResourceAllocationInfo(m_nodeMask, 1, &texture->Resource.Get()->GetDesc());

	cudaExternalMemoryHandleDesc cuExtmemHandleDesc{};
	cuExtmemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
	cuExtmemHandleDesc.handle.win32.handle = sharedHandle;
	cuExtmemHandleDesc.size = texAllocInfo.SizeInBytes;
	cuExtmemHandleDesc.flags = cudaExternalMemoryDedicated;
	checkCudaErrors(cudaImportExternalMemory(&m_externalMemoryCubemap, &cuExtmemHandleDesc));

	cudaExternalMemoryMipmappedArrayDesc cuExtmemMipDesc{};
	cuExtmemMipDesc.extent = make_cudaExtent(texture->Width, texture->Height, 0);
	cuExtmemMipDesc.formatDesc = cudaCreateChannelDesc<float4>();
	cuExtmemMipDesc.numLevels = 1;
	cuExtmemMipDesc.flags = cudaArrayCubemap;

	cudaMipmappedArray_t cuMipArray{};
	checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(&cuMipArray, m_externalMemoryCubemap, &cuExtmemMipDesc));

	cudaArray_t cuArray{};
	checkCudaErrors(cudaGetMipmappedArrayLevel(&cuArray, cuMipArray, 0));

	// allocate array and copy image data
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindFloat);
	cudaArray* cu_3darray;
	checkCudaErrors(cudaMalloc3DArray(&cu_3darray, &channelDesc, make_cudaExtent(width, width, num_faces), cudaArrayCubemap));
	cudaMemcpy3DParms myparms = { 0 };
	myparms.srcPos = make_cudaPos(0, 0, 0);
	myparms.dstPos = make_cudaPos(0, 0, 0);
	myparms.srcPtr = make_cudaPitchedPtr(h_data, width * sizeof(float), width, width);
	myparms.dstArray = cu_3darray;
	myparms.extent = make_cudaExtent(width, width, num_faces);
	myparms.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&myparms));

	cudaResourceDesc cuResDesc{};
	cuResDesc.resType = cudaResourceTypeArray;
	cuResDesc.res.array.array = cu_3darray;

	cudaTextureDesc cuTexDesc{};
	cuTexDesc.sRGB = true;
	cuTexDesc.normalizedCoords = true;
	cuTexDesc.filterMode = cudaFilterModeLinear;
	cuTexDesc.addressMode[0] = cudaAddressModeWrap;
	cuTexDesc.addressMode[1] = cudaAddressModeWrap;
	cuTexDesc.addressMode[2] = cudaAddressModeWrap;
	cuTexDesc.readMode = cudaReadModeElementType;

	// where cudaSurfaceObject_t cuSurface{};
	checkCudaErrors(cudaCreateTextureObject(& (texture->cuTexture), &cuResDesc, &cuTexDesc, NULL));
}

void CudaManager::InitCudaDatas()
{
	//initCUDAmemoryTriMesh(dev_triangle_p);
	prepCUDAscene();
}

void CudaManager::SetEnvmap()
{
	SetEnvironment(textureenv->cuSurface);
}

CudaManager::~CudaManager()
{
	void CudaFree(float* dev_triangle_p);
}