#include <Precompiled.h>
#include <ResourceBindingModule.h>
#include <TextureHelper.h>

D3DModules::ResourceBindingModule::ResourceBindingModule(ID3D12Device* d3dDevice):
	ComputeSub(this)
{
	m_d3dDevice = d3dDevice;
}

PassConstants& D3DModules::ResourceBindingModule::GetPassConstants()
{
	return passConstants;
}

ObjectConstants& D3DModules::ResourceBindingModule::GetObjectConstants()
{
	return objConstants;
}

void D3DModules::ResourceBindingModule::UpdatePassConstants(FrameResource* currFrameResource, int i)
{
	currFrameResource->passCB->CopyData(i, passConstants);
}

void D3DModules::ResourceBindingModule::UpdateObjectConstants(FrameResource* currFrameResource)
{
	//currFrameResource->passCB->CopyData(0, objConstants);
}

void D3DModules::ResourceBindingModule::BindPassConstants(ID3D12GraphicsCommandList* commandList, FrameResource* currFrameResource, int i)
{
	auto passCB = currFrameResource->passCB->Resource();
	D3D12_GPU_VIRTUAL_ADDRESS passCBAddress = passCB->GetGPUVirtualAddress() + i * passCBByteSize;
	commandList->SetGraphicsRootConstantBufferView(2, passCBAddress);
}

void D3DModules::ResourceBindingModule::CreateBuffer(uint sizeInBytes)
{

	//m_d3dDevice->CreateCommittedResource(
	//	&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
	//	D3D12_HEAP_FLAG_NONE,
	//	&CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
	//	D3D12_RESOURCE_STATE_COPY_DEST,
	//	nullptr
	//);
}

void D3DModules::ComputeSubmodule::BuildPostProcessRootSignature()
{
	// 1. Create tables
	// --------------------------------------
	//ʹ����������
	//����SRV����������Ϊ������0
	CD3DX12_DESCRIPTOR_RANGE srvTable;
	srvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV,	//����������SRV
		1,	//������������
		0);	//���������󶨵ļĴ����ۺ�

	//����UAV����������Ϊ������1
	CD3DX12_DESCRIPTOR_RANGE uavTable;
	uavTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV,	//����������SRV
		1,	//������������
		0);	//���������󶨵ļĴ����ۺ�

	// 2. Create Root parameter
	// --------------------------------------
	//������������������������������������
	CD3DX12_ROOT_PARAMETER slotRootParameter[3];
	// Ϊ����߳������ܣ����ձ��Ƶ���ɸߵ�����д������
	slotRootParameter[0].InitAsConstants(12, 0);//12���������Ĵ����ۺ�Ϊ0
	slotRootParameter[1].InitAsDescriptorTable(1, &srvTable);//Range����Ϊ1
	slotRootParameter[2].InitAsDescriptorTable(1, &uavTable);//Range����Ϊ1

	auto staticSamplers = TextureHelper::GetStaticSamplers();	//��þ�̬����������

	// 3. Create Root signature
	// --------------------------------------
	//��ǩ����һ�����������
	CD3DX12_ROOT_SIGNATURE_DESC rootSig(3, //������������
		slotRootParameter,	//������ָ��
		0,					//��̬������������0
		nullptr,			//��̬������ָ��Ϊ��
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	//�õ����Ĵ�����������һ����ǩ�����ò�λָ��һ�������е�������������������������
	ComPtr<ID3DBlob> serializedRootSig = nullptr;
	ComPtr<ID3DBlob> errorBlob = nullptr;
	HRESULT hr = D3D12SerializeRootSignature(&rootSig, D3D_ROOT_SIGNATURE_VERSION_1, &serializedRootSig, &errorBlob);

	if (errorBlob != nullptr)
	{
		OutputDebugStringA((char*)errorBlob->GetBufferPointer());
	}
	ThrowIfFailed(hr);

	m_CSRootSignatures["PostProcessing"] = nullptr;
	ThrowIfFailed(main->m_d3dDevice->CreateRootSignature(
		0,
		serializedRootSig->GetBufferPointer(),
		serializedRootSig->GetBufferSize(),
		IID_PPV_ARGS(&m_CSRootSignatures["PostProcessing"])));
}

void D3DModules::ComputeSubmodule::BuildPolygonSHRootSignature()
{
	// 1. Create tables
	// --------------------------------------
	//����UAV����������Ϊ������1
	CD3DX12_DESCRIPTOR_RANGE uavTable;
	uavTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV,	//����������SRV
		1,	//������������
		0);	//���������󶨵ļĴ����ۺ�

	// 2. Create Root parameter
	// --------------------------------------
	//������������������������������������
	CD3DX12_ROOT_PARAMETER slotRootParameter[3];
	// Ϊ����߳������ܣ����ձ��Ƶ���ɸߵ�����д������
	slotRootParameter[0].InitAsConstants(12, 0);//12���������Ĵ����ۺ�Ϊ0
	slotRootParameter[1].InitAsShaderResourceView(0, 1);//Range����Ϊ1
	slotRootParameter[2].InitAsDescriptorTable(1, &uavTable);//Range����Ϊ1

	auto staticSamplers = TextureHelper::GetStaticSamplers();	//��þ�̬����������

	// 3. Create Root signature
	// --------------------------------------
	//��ǩ����һ�����������
	CD3DX12_ROOT_SIGNATURE_DESC rootSig(3, //������������
		slotRootParameter,	//������ָ��
		0,					//��̬������������0
		nullptr,			//��̬������ָ��Ϊ��
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	//�õ����Ĵ�����������һ����ǩ�����ò�λָ��һ�������е�������������������������
	ComPtr<ID3DBlob> serializedRootSig = nullptr;
	ComPtr<ID3DBlob> errorBlob = nullptr;
	HRESULT hr = D3D12SerializeRootSignature(&rootSig, D3D_ROOT_SIGNATURE_VERSION_1, &serializedRootSig, &errorBlob);

	if (errorBlob != nullptr)
	{
		OutputDebugStringA((char*)errorBlob->GetBufferPointer());
	}
	ThrowIfFailed(hr);

	m_CSRootSignatures["PolygonSH"] = nullptr;
	ThrowIfFailed(main->m_d3dDevice->CreateRootSignature(
		0,
		serializedRootSig->GetBufferPointer(),
		serializedRootSig->GetBufferSize(),
		IID_PPV_ARGS(&m_CSRootSignatures["PolygonSH"])));
}


// ---------------------------------------
// PSO
// ---------------------------------------

void D3DModules::ComputeInstance::CreateComputePSO(ID3D12Device* d3dDevice)
{
	D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
	psoDesc.pRootSignature = m_Rootsignature;
	psoDesc.CS =
	{
		reinterpret_cast<BYTE*>(m_ComputeShader->csBytecode->GetBufferPointer()),
		m_ComputeShader->csBytecode->GetBufferSize()
	};
	psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
	ThrowIfFailed(d3dDevice->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_PSO)));
}

void D3DModules::ComputeSubmodule::CreateComputeInstance(std::string name, std::string signature, std::wstring shadername, std::string csfunc)
{
	m_ComputeInstances[name] = std::make_unique<ComputeInstance>(
		main->m_d3dDevice,
		shadername,
		m_CSRootSignatures[signature].Get(),	// Signature
		csfunc);
}