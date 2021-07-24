#include <Precompiled.h>
#include <WorkSubmissionModule.h>

ID3D12CommandQueue* D3DModules::WorkSubmissionModule::CreateCommandQueue(std::string name, D3D12_COMMAND_LIST_TYPE type)
{
	// If the name has already exists, return
	if (m_CmdQueues.find(name) != m_CmdQueues.end())
	{
		throw XGException("Create Command Queue failed :: Duplicate Name!");
	}
	// Write the descriptor
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.Type = type;
	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

	// Create new Command Queue
	m_CmdQueues[name] = nullptr;
	DXCall(m_d3dDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&(m_CmdQueues[name]))));

	return GetCommandQueue(name);
}


ID3D12CommandQueue* D3DModules::WorkSubmissionModule::GetCommandQueue(std::string name)
{
	return m_CmdQueues[name].Get();
}

ID3D12CommandAllocator* D3DModules::WorkSubmissionModule::CreateCommandListAllocator(std::string name, D3D12_COMMAND_LIST_TYPE type)
{
	// If the name has already exists, return
	if (m_CmdQueues.find(name) != m_CmdQueues.end())
	{
		throw XGException("Create Command List Allocator failed :: Duplicate Name!");
	}
	// Create new Comand List Allocator
	m_CmdListAllocators[name] = nullptr;
	DXCall(m_d3dDevice->CreateCommandAllocator(
		D3D12_COMMAND_LIST_TYPE_DIRECT,
		IID_PPV_ARGS(m_CmdListAllocators[name].GetAddressOf())));

	return GetListAllocator(name);
}

ID3D12CommandAllocator* D3DModules::WorkSubmissionModule::GetListAllocator(std::string name)
{
	return m_CmdListAllocators[name].Get();
}

ID3D12GraphicsCommandList* D3DModules::WorkSubmissionModule::CreateCommandList(std::string name, D3D12_COMMAND_LIST_TYPE type, std::string allocator)
{
	// If the name has already exists, return
	if (m_CmdLists.find(name) != m_CmdLists.end())
	{
		throw XGException("Create Command List failed :: Duplicate Name!");
	}
	// Create new Comand List Allocator
	m_CmdLists[name] = nullptr;
	DXCall(m_d3dDevice->CreateCommandList(
		0,								//									掩码值为0，单GPU
		type,							//									命令列表类型
		GetListAllocator(allocator),	// Associated command allocator		命令分配器接口指针
		nullptr,						// Initial PipelineStateObject		流水线状态对象PSO，这里不绘制，所以空指针
		IID_PPV_ARGS(m_CmdLists[name].GetAddressOf())));	//				返回创建的命令列表

	return GetCommandList(name);
}

ID3D12GraphicsCommandList* D3DModules::WorkSubmissionModule::GetCommandList(std::string name)
{
	return m_CmdLists[name].Get();
}