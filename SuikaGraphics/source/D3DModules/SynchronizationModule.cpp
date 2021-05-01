#include <SynchronizationModule.h>
#include <memory>

void D3DModules::SynchronizationModule::BuildFrameResources(RenderItemManager* RIManager, int vertex_num)
{
	for (int i = 0; i < frameResourcesCount; i++)
	{
		FrameResourcesArray.push_back(std::make_unique<FrameResource>(
			m_d3dDevice,
			1,     //passCount
			(UINT)RIManager->mAllRitems.size(),
			(UINT)RIManager->mMaterials.size(),
			vertex_num));	//objCount
	}
}

void D3DModules::SynchronizationModule::StartUpdate()
{
	// Cycle through the circular frame resource array.
	currFrameResourcesIndex = (currFrameResourcesIndex + 1) % frameResourcesCount;
	mCurrFrameResource = FrameResourcesArray[currFrameResourcesIndex].get();
	// Has the GPU finished processing the commands of the current frame resource.
	// If not, wait until the GPU has completed commands up to this fence point.
	//���GPU��Χ��ֵС��CPU��Χ��ֵ����CPU�ٶȿ���GPU������CPU�ȴ�
	if (mCurrFrameResource->fenceCPU != 0 && m_fence->GetCompletedValue() < mCurrFrameResource->fenceCPU)
	{
		HANDLE eventHandle = CreateEvent(nullptr, false, false, L"FenceSetDone");
		ThrowIfFailed(m_fence->SetEventOnCompletion(mCurrFrameResource->fenceCPU, eventHandle));
		WaitForSingleObject(eventHandle, INFINITE);
		CloseHandle(eventHandle);
	}
}

void D3DModules::SynchronizationModule::EndUpdate(ID3D12CommandQueue* m_CommandQueue)
{
	//// Wait until frame commands are complete. This waiting is
	//// inefficient and is done for simplicity. Later we will show how to
	//// organize our rendering code so we do not have to wait per frame.
	//FlushCmdQueue();
	
	if (!EndUpdateDelegates->empty())
	{
		(*EndUpdateDelegates)();
		EndUpdateDelegates->clear();
	}

	// Advance the fence value to mark commands up to this fence point.
	mCurrFrameResource->fenceCPU = ++mCurrentFence;
	// Add an instruction to the command queue to set a new fence point.
	// Because we are on the GPU timeline, the new fence point won��t be
	// set until the GPU finishes processing all the commands prior to
	// this Signal().
	m_CommandQueue->Signal(m_fence.Get(), mCurrentFence);
	// Note that GPU could still be working on commands from previous
	// frames, but that is okay, because we are not touching any frame
	// resources associated with those frames.
}

void D3DModules::SynchronizationModule::FlushCmdQueue(ID3D12CommandQueue* m_CommandQueue)
{
	mCurrentFence++;	//CPU��������رպ󣬽���ǰΧ��ֵ+1
	m_CommandQueue->Signal(m_fence.Get(), mCurrentFence);	//��GPU������CPU���������󣬽�fence�ӿ��е�Χ��ֵ+1����fence->GetCompletedValue()+1
	Debug::LogSystem(QString::number(m_fence->GetCompletedValue()) + QString(":") + QString::number(mCurrentFence));
	if (m_fence->GetCompletedValue() < mCurrentFence)	//���С�ڣ�˵��GPUû�д�������������
	{
		HANDLE eventHandle = CreateEvent(nullptr, false, false, L"FenceSetDone");	//�����¼�
		m_fence->SetEventOnCompletion(mCurrentFence, eventHandle);//��Χ���ﵽmCurrentFenceֵ����ִ�е�Signal����ָ���޸���Χ��ֵ��ʱ������eventHandle�¼�
		WaitForSingleObject(eventHandle, INFINITE);//�ȴ�GPU����Χ���������¼���������ǰ�߳�ֱ���¼�������ע���Enent���������ٵȴ���
							   //���û��Set��Wait���������ˣ�Set��Զ������ã�����Ҳ��û�߳̿��Ի�������̣߳�
		CloseHandle(eventHandle);
	}
}