#pragma once
#include <string>
#include <Utility.h>

using Microsoft::WRL::ComPtr;

namespace D3DModules
{
	class WorkSubmissionModule
	{
	public:
		WorkSubmissionModule(ID3D12Device* device) :m_d3dDevice(device) {}
		void CreateCommandList();



		ID3D12CommandQueue* CreateCommandQueue(std::string name, D3D12_COMMAND_LIST_TYPE type);
		ID3D12CommandQueue* GetCommandQueue(std::string name);

		ID3D12CommandAllocator* CreateCommandListAllocator(std::string name, D3D12_COMMAND_LIST_TYPE type);
		ID3D12CommandAllocator* GetListAllocator(std::string name);

		ID3D12GraphicsCommandList* CreateCommandList(std::string name, D3D12_COMMAND_LIST_TYPE type, std::string allocator);
		ID3D12GraphicsCommandList* GetCommandList(std::string name);

		void ResetCommandList(ID3D12GraphicsCommandList* commandList, ID3D12CommandAllocator* allcator, ID3D12PipelineState* pipelineState)
		{
			DXCall(allcator->Reset());//重复使用记录命令的相关内存
			DXCall(commandList->Reset(allcator, pipelineState));//复用命令列表及其内存
		}

		void ResetPipelineState(ID3D12GraphicsCommandList* commandList, ID3D12PipelineState* state)
		{

		}

	private:
		std::unordered_map<std::string, ComPtr<ID3D12CommandAllocator>>			m_CmdListAllocators;
		std::unordered_map<std::string, ComPtr<ID3D12CommandQueue>>				m_CmdQueues;
		std::unordered_map<std::string, ComPtr<ID3D12GraphicsCommandList>>		m_CmdLists;

		ID3D12Device* m_d3dDevice;
		ID3D12CommandAllocator* p_CurrAllocator;
		ID3D12PipelineState* p_CurrInitialState;
	};
}