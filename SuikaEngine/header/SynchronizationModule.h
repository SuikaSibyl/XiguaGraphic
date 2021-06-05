#pragma once
#include <string>
#include <Utility.h>
#include <PipelineSetting.h>
#include <FrameResources.h>
#include <RenderItemManagment.h>
#include <Delegate.h>

using Microsoft::WRL::ComPtr;

namespace D3DModules
{
	class SynchronizationModule
	{
	public:
		int currFrameResourcesIndex = 0;
		FrameResource* mCurrFrameResource = nullptr;

		SynchronizationModule(ID3D12Device* device) :m_d3dDevice(device) 
		{
			EndUpdateDelegates = new Delegate::CMultiDelegate<void>();
		}

		void CreateFence()
		{
			DXCall(m_d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_fence)));
		}
		// Init Frame Resource,must after all render items pushed;
		void BuildFrameResources(RenderItemManager* RIManager, int vertex_num);

		void StartUpdate();

		void EndUpdate(ID3D12CommandQueue* m_CommandQueue);

		void FlushCmdQueue(ID3D12CommandQueue* m_CommandQueue);

		void SetMainCommandQueue(ID3D12CommandQueue* main)
		{
			m_MainCommandQueue = main;
		}

		void SynchronizeMainQueue()
		{
			FlushCmdQueue(m_MainCommandQueue);
		}

		void ExecuteCommandList(ID3D12GraphicsCommandList* commandList)
		{
			ThrowIfFailed(commandList->Close());
			ID3D12CommandList* cmdLists[] = { commandList };
			m_MainCommandQueue->ExecuteCommandLists(_countof(cmdLists), cmdLists);
		}

		ID3D12CommandQueue* GetMainQueue()
		{
			return m_MainCommandQueue;
		}

		template< typename T>
		void AddEndUpdateListening(T func)
		{
			(*EndUpdateDelegates) += Delegate::newDelegate(func);
		}

		template< typename T, typename F>
		void AddEndUpdateListeningMem(T* _object, F func)
		{
			(*EndUpdateDelegates) += Delegate::newDelegate(_object, func);
		}

		ComPtr<ID3D12Fence>	m_fence;
		UINT64	mCurrentFence = 0;	    //Initial CPU Fence = 0

	private:
		std::vector<std::unique_ptr<FrameResource>> FrameResourcesArray;
		ID3D12Device* m_d3dDevice;

		ID3D12CommandQueue* m_MainCommandQueue;

		Delegate::CMultiDelegate<void>* EndUpdateDelegates;
	};
}