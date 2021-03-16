#include "QDirect3D12Widget.h"
#include <DirectXColors.h>

#include <array>
#include <QDebug>
#include <QEvent>
#include <QWheelEvent>
#include <DirectXMath.h>
#include <QTime>

#include <Shader.h>
#include <UploadBuffer.h>
#include <GeometryGenerator.h>
#include <SuikaGraphics.h>
#include <Singleton.h>
#include <Material.h>
#include <Texture.h>
#include <TextureHelper.h>

using Microsoft::WRL::ComPtr;
using namespace DirectX;
using Geometry::RenderItem;

constexpr int FPS_LIMIT = 120.0f;
constexpr int MS_PER_FRAME = (int)((1.0f / FPS_LIMIT) * 1000.0f);

#pragma region LIFECYCLE

/// <summary>
/// LIFECYCLE :: Initialization (First Show)
/// </summary>
bool QDirect3D12Widget::Initialize()
{
	// First resize, will calc mProj
	onResize();
	// Initialize Direct3D
	InitDirect3D();

	try
	{
		m_CommandList->Reset(m_DirectCmdListAlloc.Get(), nullptr);

		BuildRootSignature2();
		BuildShadersAndInputLayout();

		// Material
		BuildMaterial();
		// Geometry Things
		BuildMultiGeometry();
		BuildLandGeometry();
		BuildLakeGeometry();
		BuildLights();
		BuildTexture();
		
		// Init Frame Resource
		// must after all render items pushed;
		BuildFrameResources();

		//BuildDescriptorHeaps();
		//BuildConstantBuffers();
		BuildPSO();

		ThrowIfFailed(m_CommandList->Close());
		ID3D12CommandList* cmdLists[] = { m_CommandList.Get() };
		m_CommandQueue->ExecuteCommandLists(_countof(cmdLists), cmdLists);

		// Wait until initialization is complete.
		FlushCmdQueue();

		// Releasde uploader resource
		RIManager.DisposeAllUploaders();
	}
	catch (HrException& e)
	{
		MessageBox(nullptr, e.ToLPCWSTR(), L"HR  Failed", MB_OK);
		return 0;
	}

	// Start FrameLoop
	connect(&m_qTimer, &QTimer::timeout, this, &QDirect3D12Widget::onFrame);
	// Start Timer
	m_tGameTimer.Reset();

	return true;
}

float GetWave(float x, float z, float t) {
	return 0.04f * (z * sinf(0.05f * x * t) + x * cosf(0.03f * z * t));
}

/// <summary>
/// LIFECYCLE :: Update (Before Draw)
/// </summary>
void QDirect3D12Widget::Update()
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

	ObjectConstants objConstants;
	PassConstants passConstants;

	MainCamera.Update();
	XMMATRIX view = MainCamera.GetViewMatrix();
	XMStoreFloat4x4(&mView, view);
	XMMATRIX proj = XMLoadFloat4x4(&mProj);

	XMMATRIX viewProj = view * proj;
	// Update the constant buffer with the latest worldViewProj matrix.
	passConstants.gTime = m_tGameTimer.GetTotalTime();
	passConstants.eyePos = MainCamera.GetPosition();
	passConstants.light[0] = *RIManager.mLights["mainLit"];
	XMStoreFloat4x4(&passConstants.viewProj, XMMatrixTranspose(viewProj));
	mCurrFrameResource->passCB->CopyData(0, passConstants);

	// Update object Index
	for (auto& e : RIManager.mAllRitems)
	{
		if (e->NumFramesDirty > 0)
		{
			XMMATRIX w = XMLoadFloat4x4(&e->World);
			//XMMATRIX��ֵ��XMFLOAT4X4
			XMStoreFloat4x4(&objConstants.world, XMMatrixTranspose(w));
			//�����ݿ�����GPU����
			mCurrFrameResource->objCB->CopyData(e->ObjCBIndex, objConstants);

			e->NumFramesDirty--;
		}
	}

	// Update Material
	UploadBuffer<MaterialConstants>* currMaterialCB = mCurrFrameResource -> materialCB.get();
	for (auto& e : RIManager.mMaterials)
	{
		// Only update the cbuffer data if the constants have changed.If
		// the cbuffer data changes, it needs to be updated for each
		// FrameResource
		Material* mat = e.second.get();
		if (mat->NumFramesDirty > 0)
		{
			XMMATRIX matTransform = XMLoadFloat4x4(&mat -> MatTransform);
			MaterialConstants matConstants;
			matConstants.DiffuseAlbedo = mat->DiffuseAlbedo;
			matConstants.FresnelR0 = mat->FresnelR0;
			matConstants.Roughness = mat->Roughness;

			currMaterialCB->CopyData(mat->MatCBIndex, matConstants);
			// Next FrameResource need to be updated too.
			mat->NumFramesDirty--;
		}
	}

	// Update Wave
	auto currWavesVB = mCurrFrameResource -> dynamicVB.get();
	vector<Vertex>& vertices = wave->helper.GetVertices();
	for (int i = 0; i < vertex_num; i++)
	{
		vertices[i].Pos.y = GetWave(vertices[i].Pos.x, vertices[i].Pos.z, GameTimer::TotalTime());
	}
	wave->helper.CalcNormal();
	for (int i = 0; i < vertex_num; i++)
	{
		currWavesVB->CopyData(i, vertices[i]);
	}
	RIManager.geometries["lakeGeo"]->VertexBufferGPU = currWavesVB -> Resource();
}

/// <summary>
/// LIFECYCLE :: Draw Stuff
/// </summary>
void QDirect3D12Widget::Draw()
{
	//�����������������cmdAllocator�������б�cmdList��Ŀ��������������б���������ڴ档
	// Reuse the memory associated with command recording.
	// We can only reset when the associated command lists have finished
	// execution on the GPU.
	auto currCmdAllocator = mCurrFrameResource->cmdAllocator;
	DXCall(currCmdAllocator->Reset());//�ظ�ʹ�ü�¼���������ڴ�
	DXCall(m_CommandList->Reset(currCmdAllocator.Get(), mPSO.Get()));//���������б����ڴ�

	// Indicate a state transition on the resource usage.
	//�������ǽ���̨������Դ�ӳ���״̬ת������ȾĿ��״̬����׼������ͼ����Ⱦ����
	UINT& ref_mCurrentBackBuffer = mCurrentBackBuffer;
	m_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_SwapChainBuffer[ref_mCurrentBackBuffer].Get(),//ת����ԴΪ��̨��������Դ
		D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));//�ӳ��ֵ���ȾĿ��ת��

	//�����������ӿںͲü����Ρ�
	m_CommandList->RSSetViewports(1, &viewPort);
	m_CommandList->RSSetScissorRects(1, &scissorRect);

	// Clear the back buffer and depth buffer.
	//Ȼ�������̨����������Ȼ�����������ֵ���������Ȼ�ö������������������ַ������ͨ��ClearRenderTargetView������ClearDepthStencilView����������͸�ֵ���������ǽ�RT��Դ����ɫ��ֵΪDarkRed�����죩��
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), ref_mCurrentBackBuffer, m_rtvDescriptorSize);
	m_CommandList->ClearRenderTargetView(rtvHandle, DirectX::Colors::DarkRed, 0, nullptr);//���RT����ɫΪ���죬���Ҳ����òü�����
	D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
	m_CommandList->ClearDepthStencilView(dsvHandle,	//DSV���������
		D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
		1.0f,	//Ĭ�����ֵ
		0,	//Ĭ��ģ��ֵ
		0,	//�ü���������
		nullptr);	//�ü�����ָ��

	// Specify the buffers we are going to render to. 
	//mCommandList->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());
	//Ȼ������ָ����Ҫ��Ⱦ�Ļ���������ָ��RTV��DSV��
	m_CommandList->OMSetRenderTargets(1,//���󶨵�RTV����
		&rtvHandle,	//ָ��RTV�����ָ��
		true,	//RTV�����ڶ��ڴ�����������ŵ�
		&dsvHandle);	//ָ��DSV��ָ��

	//ID3D12DescriptorHeap* descriptorHeaps[] = { m_cbvHeap.Get() };
	//m_CommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
	m_CommandList->SetGraphicsRootSignature(mRootSignature.Get());
	//����SRV��������
	//ע������֮���������飬����Ϊ�����ܰ���SRV��UAV������������ֻ�õ���SRV
	ID3D12DescriptorHeap* descriptorHeaps[] = { m_srvHeap.Get() };
	m_CommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

	UINT passConstSize = Utils::CalcConstantBufferByteSize(sizeof(PassConstants));
	auto passCB = mCurrFrameResource->passCB-> Resource();
	m_CommandList->SetGraphicsRootConstantBufferView(1, passCB-> GetGPUVirtualAddress());
	// Deprecated: use descriptor table
	//		int passCbvIndex = (int)mMultiGeo->RenderItems.size() * frameResourcesCount + currFrameResourcesIndex;
	//		auto handle = CD3DX12_GPU_DESCRIPTOR_HANDLE(m_cbvHeap->GetGPUDescriptorHandleForHeapStart());
	//		handle.Offset(passCbvIndex, m_cbv_srv_uavDescriptorSize);
	//		m_CommandList->SetGraphicsRootDescriptorTable(1, //����������ʼ���� handle);

	DrawRenderItems();

	// Indicate a state transition on the resource usage.
	// �ȵ���Ⱦ��ɣ�����Ҫ����̨��������״̬�ĳɳ���״̬��ʹ��֮���Ƶ�ǰ̨��������ʾ�����ˣ��ر������б��ȴ�����������С�
	m_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_SwapChainBuffer[ref_mCurrentBackBuffer].Get(),
		D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));//����ȾĿ�굽����
	// �������ļ�¼�ر������б� 
	ThrowIfFailed(m_CommandList->Close());

	// Add the command list to the queue for execution.
	//��CPU�����׼���ú���Ҫ����ִ�е������б����GPU��������С�ʹ�õ���ExecuteCommandLists������
	ID3D12CommandList* commandLists[] = { m_CommandList.Get() };//���������������б�����
	m_CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);//������������б����������

	// swap the back and front buffers
	ThrowIfFailed(m_SwapChain->Present(0, 0));
	ref_mCurrentBackBuffer = (ref_mCurrentBackBuffer + 1) % 2;

	//// Wait until frame commands are complete. This waiting is
	//// inefficient and is done for simplicity. Later we will show how to
	//// organize our rendering code so we do not have to wait per frame.
	//FlushCmdQueue();

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
void QDirect3D12Widget::DrawRenderItems()
{
	//������ָ������ת������ָͨ������
	std::vector<RenderItem*> ritems;
	for (auto& e : RIManager.mAllRitems)
		ritems.push_back(e.get());

	auto objectCB = mCurrFrameResource->objCB->Resource();
	INT objCBByteSize = Utils::CalcConstantBufferByteSize(sizeof(ObjectConstants));
	INT matCBByteSize = Utils::CalcConstantBufferByteSize(sizeof(MaterialConstants));

	//������Ⱦ������
	for (size_t i = 0; i < ritems.size(); i++)
	{
		auto ritem = ritems[i];

		m_CommandList->IASetVertexBuffers(0, 1, &ritem->Geo->VertexBufferView());
		m_CommandList->IASetIndexBuffer(&ritem->Geo->IndexBufferView());
		m_CommandList->IASetPrimitiveTopology(ritem->PrimitiveType);

		CD3DX12_GPU_DESCRIPTOR_HANDLE tex( m_srvHeap -> GetGPUDescriptorHandleForHeapStart());
		tex.Offset(0, m_cbv_srv_uavDescriptorSize);
		m_CommandList->SetGraphicsRootDescriptorTable(3, tex);

		//���ø�������,��������������Դ��
		auto objCB = mCurrFrameResource->objCB->Resource();
		auto objCBAddress = objCB->GetGPUVirtualAddress();
		objCBAddress += ritem->ObjCBIndex * objCBByteSize;
		m_CommandList->SetGraphicsRootConstantBufferView(0,//�Ĵ����ۺ�
			objCBAddress);//����Դ��ַ
	// Deprecated: use descriptor table
	//		UINT objCbvIndex = currFrameResourcesIndex * (UINT)mMultiGeo->RenderItems.size() + ritem->ObjCBIndex;
	//		auto handle = CD3DX12_GPU_DESCRIPTOR_HANDLE(m_cbvHeap->GetGPUDescriptorHandleForHeapStart());
	//		handle.Offset(objCbvIndex, m_cbv_srv_uavDescriptorSize);
	//		m_CommandList->SetGraphicsRootDescriptorTable(0, //����������ʼ����
	//			handle);

		auto matCB = mCurrFrameResource->materialCB->Resource();
		D3D12_GPU_VIRTUAL_ADDRESS matCBAddress = matCB->GetGPUVirtualAddress();
		matCBAddress += ritem->material->MatCBIndex * matCBByteSize;
		m_CommandList->SetGraphicsRootConstantBufferView(2, matCBAddress);

		//���ƶ��㣨ͨ���������������ƣ�
		m_CommandList->DrawIndexedInstanced(ritem->IndexCount, //ÿ��ʵ��Ҫ���Ƶ�������
			1,	//ʵ��������
			ritem->StartIndexLocation,	//��ʼ����λ��
			ritem->BaseVertexLocation,	//��������ʼ������ȫ�������е�λ��
			0);	//ʵ�����ĸ߼���������ʱ����Ϊ0
	}
}
#pragma endregion

QDirect3D12Widget::QDirect3D12Widget(QWidget* parent)
	: QWidget(parent)
	, m_hWnd(reinterpret_cast<HWND>(winId()))
	, m_bDeviceInitialized(false)
	, m_bRenderActive(false)
	, m_bStarted(false)
	, m_tGameTimer(Singleton<GameTimer>::get_instance())
{
	// Set palette
	QPalette pal = palette();
	pal.setColor(QPalette::Window, Qt::black);
	setAutoFillBackground(true);
	setPalette(pal);

	setFocusPolicy(Qt::StrongFocus);
	setAttribute(Qt::WA_NativeWindow);

	// Setting these attributes to our widget and returning null on paintEngine event
	// tells Qt that we'll handle all drawing and updating the widget ourselves.
	setAttribute(Qt::WA_PaintOnScreen);
	setAttribute(Qt::WA_NoSystemBackground);

	MainCamera.m_pInputSystem = &InputSys;
	MainCamera.m_pD3dWidget = this;
	MainCamera.Init();
	MainCamera.SetMouseModeFocus();
}

QDirect3D12Widget::~QDirect3D12Widget() {}

/// <summary>
/// LIFECYCLE :: Release
/// </summary>
void QDirect3D12Widget::Release()
{
	m_bDeviceInitialized = false;
	disconnect(&m_qTimer, &QTimer::timeout, this, &QDirect3D12Widget::onFrame);
	m_qTimer.stop();
}

/// <summary>
/// LIFECYCLE :: Initialization
/// </summary>
void QDirect3D12Widget::Run()
{
	m_qTimer.start(MS_PER_FRAME);
	m_bRenderActive = m_bStarted = true;
}

/// <summary>
/// LIFECYCLE :: Pause
/// </summary>
void QDirect3D12Widget::PauseFrames()
{
	if (!m_qTimer.isActive() || !m_bStarted) return;

	disconnect(&m_qTimer, &QTimer::timeout, this, &QDirect3D12Widget::onFrame);
	m_qTimer.stop();
	m_tGameTimer.Stop();
	m_bRenderActive = false;
}

/// <summary>
/// LIFECYCLE :: Continue
/// </summary>
void QDirect3D12Widget::ContinueFrames()
{
	if (m_qTimer.isActive() || !m_bStarted) return;

	connect(&m_qTimer, &QTimer::timeout, this, &QDirect3D12Widget::onFrame);
	m_qTimer.start(MS_PER_FRAME);
	m_tGameTimer.Start();
	m_bRenderActive = true;
}

void QDirect3D12Widget::BuildDescriptorHeaps()
{
	UINT objectCount = (UINT)RIManager.mAllRitems.size();//�����ܸ���������ʵ����

	// ����Descriptor heap, ������CBV
	// ���ȴ���cbv��
	D3D12_DESCRIPTOR_HEAP_DESC cbvHeapDesc;
	cbvHeapDesc.NumDescriptors = (objectCount + 1) * frameResourcesCount;
	cbvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	cbvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	cbvHeapDesc.NodeMask = 0;
	ThrowIfFailed(m_d3dDevice->CreateDescriptorHeap(&cbvHeapDesc, IID_PPV_ARGS(m_cbvHeap.GetAddressOf())));
}

void QDirect3D12Widget::BuildConstantBuffers()
{
	// Create Object CBV
	// ----------------------
	UINT objectCount = (UINT)RIManager.mAllRitems.size();//�����ܸ���������ʵ����
	UINT objCBByteSize = Utils::CalcConstantBufferByteSize(sizeof(ObjectConstants));
	//D3D12_GPU_VIRTUAL_ADDRESS objCbAddress = mObjectCB->Resource()->GetGPUVirtualAddress();
	//// Offset to the ith object constant buffer in the buffer.
	//// Here our i = 0.
	for (int frameIndex = 0; frameIndex < frameResourcesCount; frameIndex++)
	{
		for (int i = 0; i < objectCount; i++)
		{
			D3D12_GPU_VIRTUAL_ADDRESS objCB_Address;
			objCB_Address = FrameResourcesArray[frameIndex]->objCB->Resource()->GetGPUVirtualAddress();
			objCB_Address += i * objCBByteSize;//�������ڳ����������еĵ�ַ
			int heapIndex = objectCount * frameIndex + i;	//CBV���е�CBVԪ������
			auto handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_cbvHeap->GetCPUDescriptorHandleForHeapStart());//���CBV���׵�ַ
			handle.Offset(heapIndex, m_cbv_srv_uavDescriptorSize);	//CBV�����CBV���е�CBVԪ�ص�ַ��
			//����CBV������
			D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
			cbvDesc.BufferLocation = objCB_Address;
			cbvDesc.SizeInBytes = objCBByteSize;
			m_d3dDevice->CreateConstantBufferView(&cbvDesc, handle);
		}
	}

	// Create Pass CBV
	// ----------------------
	UINT passCBByteSize = Utils::CalcConstantBufferByteSize(sizeof(PassConstants));

	for (int frameIndex = 0; frameIndex < frameResourcesCount; frameIndex++)
	{
		D3D12_GPU_VIRTUAL_ADDRESS passCbAddress;
		passCbAddress = FrameResourcesArray[frameIndex]->passCB->Resource()->GetGPUVirtualAddress();;
		// Offset to the ith object constant buffer in the buffer.
		// Here our i = 0.
		int passCbElementIndex = 0;
		passCbAddress += passCbElementIndex * passCBByteSize;

		int heapIndex = objectCount * frameResourcesCount + frameIndex;
		auto handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_cbvHeap->GetCPUDescriptorHandleForHeapStart());
		handle.Offset(heapIndex, m_cbv_srv_uavDescriptorSize);
		//����CBV������
		D3D12_CONSTANT_BUFFER_VIEW_DESC passCbvDesc;
		passCbvDesc.BufferLocation = passCbAddress;
		passCbvDesc.SizeInBytes = passCBByteSize;
		m_d3dDevice->CreateConstantBufferView(&passCbvDesc, handle);
	}
}
float GetHeight(float x, float z){
	return 0.3f * (z * sinf(0.1f * x) + x * cosf(0.1f * z));
}

void QDirect3D12Widget::BuildLandGeometry() {
	ProceduralGeometry::GeometryGenerator geoGen;
	ProceduralGeometry::GeometryGenerator::MeshData grid =
		geoGen.CreateGrid(160.0f, 160.0f, 50, 50);
	//
	// Extract the vertex elements we are interested and apply the height
	// function to each vertex. In addition, color the vertices based on
	// their height so we have sandy looking beaches, grassy low hills,
	// and snow mountain peaks.
	//
	std::vector<Geometry::Vertex> vertices(grid.Vertices.size());
	for (size_t i = 0; i < grid.Vertices.size(); ++i)
	{
		auto& p = grid.Vertices[i].Position;
		vertices[i].Pos = p;
		vertices[i].Pos.y = GetHeight(p.x, p.z);
		vertices[i].Normal = XMFLOAT3(0.0f, 0.0f, 0.0f);
		vertices[i].TexC = grid.Vertices[i].TexC;
		// Color the vertex based on its height.
		//if (vertices[i].Pos.y < -10.0f)
		//{
		//	// Sandy beach color.
		//	vertices[i].Color = XMFLOAT4(1.0f, 0.96f, 0.62f,
		//		1.0f);
		//}
		//else if (vertices[i].Pos.y < 5.0f)
		//{
		//	 Light yellow-green.
		//	vertices[i].Color = XMFLOAT4(0.48f, 0.77f,
		//		0.46f, 1.0f);
		//}
		//else if (vertices[i].Pos.y < 12.0f)
		//{
		//	 Dark yellow-green.
		//	vertices[i].Color = XMFLOAT4(0.1f, 0.48f, 0.19f,
		//		1.0f);
		//}
		//else if (vertices[i].Pos.y < 20.0f)
		//{
		//	 Dark brown.
		//	vertices[i].Color = XMFLOAT4(0.45f, 0.39f,
		//		0.34f, 1.0f);
		//}
		//else
		//{
		//	 White snow.
		//	vertices[i].Color = XMFLOAT4(1.0f, 1.0f, 1.0f,
		//		1.0f);
		//}
	}
	std::vector<std::uint16_t> indices = grid.GetIndices16();

	MeshGeometryHelper helper(this);
	helper.PushSubmeshGeometry("grid", vertices, indices);
	helper.CalcNormal();
	RIManager.AddGeometry("landGeo", helper.CreateMeshGeometry("landGeo"));
	RenderItem* land = RIManager.AddRitem("landGeo", "grid");
	land->material = RIManager.mMaterials["grass"].get();
}

void QDirect3D12Widget::BuildLakeGeometry() 
{
	ProceduralGeometry::GeometryGenerator geoGen;
	ProceduralGeometry::GeometryGenerator::MeshData grid =
		geoGen.CreateGrid(160.0f, 160.0f, 50, 50);
	//
	// Extract the vertex elements we are interested and apply the height
	// function to each vertex. In addition, color the vertices based on
	// their height so we have sandy looking beaches, grassy low hills,
	// and snow mountain peaks.
	//
	std::vector<Geometry::Vertex> vertices(grid.Vertices.size());
	for (size_t i = 0; i < grid.Vertices.size(); ++i)
	{
		auto& p = grid.Vertices[i].Position;
		vertices[i].Pos = p;
		vertices[i].Pos.y = 0.5f;
		//vertices[i].Color = XMFLOAT4(0.26f, 0.36f, 0.92f, 1.0f);
		vertices[i].Normal = XMFLOAT3(0,0,0);
		vertices[i].TexC = grid.Vertices[i].TexC;
	}
	std::vector<std::uint16_t> indices = grid.GetIndices16();

	MeshGeometryHelper helper(this);
	helper.PushSubmeshGeometry("grid", vertices, indices);
	RIManager.AddGeometry("lakeGeo", helper.CreateMeshGeometry("lakeGeo"));
	RenderItem* lake = RIManager.AddRitem("lakeGeo", "grid");
	lake->material = RIManager.mMaterials["water"].get();

	wave = std::make_unique<Waves>(std::move(helper), vertices.size());

	vertex_num = vertices.size();
}

void QDirect3D12Widget::BuildLights()
{
	std::unique_ptr<Light> light = std::make_unique<Light>();
	light->Direction = XMFLOAT3(0, -1, 0);
	light->Position = XMFLOAT3(0, 10, 0);
	light->Strength = XMFLOAT3(1, 0.5, 1);
	RIManager.AddLight("mainLit", light);
}

void QDirect3D12Widget::BuildTexture()
{
	auto woodCrateTex = std::make_unique<Texture>();
	woodCrateTex->Name = "woodCrateTex";
	woodCrateTex->Filename = L"./Resource/Textures/WoodCrate01.dds";
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(
		m_d3dDevice.Get(), m_CommandList.Get(),
		woodCrateTex->Filename.c_str(),
		woodCrateTex->Resource, woodCrateTex -> UploadHeap));

	auto grassTex = std::make_unique<Texture>();
	grassTex->Name = "grassTexTex";
	grassTex->Filename = L"./Resource/Textures/grass.dds";
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(
		m_d3dDevice.Get(), m_CommandList.Get(),
		grassTex->Filename.c_str(),
		grassTex->Resource, grassTex->UploadHeap));

	auto bricksTex = std::make_unique<Texture>();
	bricksTex->Name = "bricksTex";
	bricksTex->Filename = L"./Resource/Textures/bricks.dds";
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(
		m_d3dDevice.Get(), m_CommandList.Get(),
		bricksTex->Filename.c_str(),
		bricksTex->Resource, bricksTex->UploadHeap));
	// Suppose the following texture resources are alreadycreated.
		// ID3D12Resource* bricksTex;
		// ID3D12Resource* stoneTex;
		// ID3D12Resource* tileTex;

	// Get pointer to the start of the heap.
	CD3DX12_CPU_DESCRIPTOR_HANDLE hDescriptor(m_srvHeap-> GetCPUDescriptorHandleForHeapStart());
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Format = woodCrateTex->Resource->GetDesc().Format;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = woodCrateTex->Resource -> GetDesc().MipLevels;
	srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
	m_d3dDevice->CreateShaderResourceView(woodCrateTex->Resource.Get(), &srvDesc, hDescriptor);

	// offset to next descriptor in heap
	hDescriptor.Offset(1, m_cbv_srv_uavDescriptorSize);
	srvDesc.Format = grassTex->Resource->GetDesc().Format;
	srvDesc.Texture2D.MipLevels = grassTex->Resource-> GetDesc().MipLevels;
	m_d3dDevice->CreateShaderResourceView(grassTex->Resource.Get(),&srvDesc, hDescriptor);

	// offset to next descriptor in heap
	hDescriptor.Offset(1, m_cbv_srv_uavDescriptorSize); srvDesc.Format = bricksTex->Resource->GetDesc().Format;
	srvDesc.Texture2D.MipLevels = bricksTex->Resource-> GetDesc().MipLevels;
	m_d3dDevice->CreateShaderResourceView(bricksTex->Resource.Get(), &srvDesc, hDescriptor);
}

void QDirect3D12Widget::BuildMaterial()
{
	std::unique_ptr<Material> grass = std::make_unique<Material>();
	grass->Name = "grass";
	grass->MatCBIndex = 0;
	grass->DiffuseAlbedo = XMFLOAT4(0.2f, 0.6f, 0.6f, 1.0f);
	grass->FresnelR0 = XMFLOAT3(0.01f, 0.01f, 0.01f);
	grass->Roughness = 0.125f;
	grass->MatCBIndex = 0;
	// This is not a good water material definition, but we do not have
	// all the rendering tools we need (transparency, environment
	// reflection), so we fake it for now.
	auto water = std::make_unique<Material>();
	water->Name = "water";
	water->MatCBIndex = 1;
	water->DiffuseAlbedo = XMFLOAT4(0.0f, 0.2f, 0.6f, 1.0f);
	water->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	water->Roughness = 0.0f;
	water->MatCBIndex = 1;

	RIManager.mMaterials["grass"] = std::move(grass);
	RIManager.mMaterials["water"] = std::move(water);
}

void QDirect3D12Widget::BuildRootSignature()
{
	// Shader programs typically require resources as input(constant
	// buffers, textures, samplers). The root signature defines the
	// resources the shader programs expect. If we think of the shader
	// programs as a function, and the input resources as function
	// parameters, then the root signature can be thought of as defining
	// the function signature.

	// root signature: define which resources will be bind to pipeline
	//				   consist of root parameters
	// root parameter: describe resources

	// 1. Create Root Parameter
	// -------------------------------------------------
	// Root parameter can be a table, root descriptor or root constants.
	// ---- 1.1 declare root parameter
	CD3DX12_ROOT_PARAMETER slotRootParameter[2];
	// ---- 1.2 create root parameter
	//			here means to Create a single descriptor table of CBVs.
	//			����������CBV����ɵ���������
	CD3DX12_DESCRIPTOR_RANGE cbvTable0;
	cbvTable0.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);
	CD3DX12_DESCRIPTOR_RANGE cbvTable1;
	cbvTable1.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 1);
	// ����������  - ���������� - ���������󶨵ļĴ����ۺ�
	// ---- 1.3 add content to the root parameter
	slotRootParameter[0].InitAsDescriptorTable(1, &cbvTable0);
	slotRootParameter[1].InitAsDescriptorTable(1, &cbvTable1);

	// 2. Create Root Signature
	// -------------------------------------------------
	// A root signature is an array of root parameters.
	//��ǩ����һ�����������
	// ---- 2.1 fill out a structure CD3DX12_ROOT_SIGNATURE_DESC
	CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(2,
		slotRootParameter, 0, nullptr,
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
	// create a root signature with a single slot which points to a
	// descriptor range consisting of a single constant buffer
	// �õ����Ĵ�����������һ����ǩ�����ò�λָ��һ�������е�������������������������
	// ---- 2.2 call CreateRootSignature
	ComPtr<ID3DBlob> serializedRootSig = nullptr;
	ComPtr<ID3DBlob> errorBlob = nullptr;
	HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc,
			D3D_ROOT_SIGNATURE_VERSION_1,
			serializedRootSig.GetAddressOf(),
			errorBlob.GetAddressOf());
	if (errorBlob != nullptr)
	{
		::OutputDebugStringA((char*)errorBlob->GetBufferPointer());
	}
	ThrowIfFailed(hr);
	ThrowIfFailed(m_d3dDevice->CreateRootSignature(
		0,
		serializedRootSig->GetBufferPointer(),
		serializedRootSig->GetBufferSize(),
		IID_PPV_ARGS(&mRootSignature)));
}

void QDirect3D12Widget::BuildRootSignature2()
{
	CD3DX12_DESCRIPTOR_RANGE srvTable;
	srvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV,	//����������
		1,	//������������
		0);	//���������󶨵ļĴ����ۺ�

	auto staticSamplers = TextureHelper::GetStaticSamplers();	//��þ�̬����������

	//������������������������������������
	CD3DX12_ROOT_PARAMETER slotRootParameter[4];
	slotRootParameter[0].InitAsConstantBufferView(0);
	slotRootParameter[1].InitAsConstantBufferView(1);
	slotRootParameter[2].InitAsConstantBufferView(2);
	slotRootParameter[3].InitAsDescriptorTable(1,//Range����
		&srvTable,	//Rangeָ��
		D3D12_SHADER_VISIBILITY_PIXEL);	//����Դֻ����������ɫ���ɶ�

	//��ǩ����һ�����������
	CD3DX12_ROOT_SIGNATURE_DESC rootSig(4, //������������
		slotRootParameter, //������ָ��
		staticSamplers.size(), 
		staticSamplers.data(),	//��̬������ָ��
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

	ThrowIfFailed(m_d3dDevice->CreateRootSignature(0,
		serializedRootSig->GetBufferPointer(),
		serializedRootSig->GetBufferSize(),
		IID_PPV_ARGS(&mRootSignature)));
}

void QDirect3D12Widget::BuildShadersAndInputLayout()
{
	mpShader = new Shader(m_d3dDevice);
}

void QDirect3D12Widget::BuildMultiGeometry()
{
	ProceduralGeometry::GeometryGenerator geoGen;
	ProceduralGeometry::GeometryGenerator::MeshData sphere = geoGen.CreateGeosphere(0.5f, 4);
	ProceduralGeometry::GeometryGenerator::MeshData cylinder = geoGen.CreateCylinder(0.5f, 0.3f, 3.0f, 20, 20);

	std::vector<Geometry::Vertex> sphere_vertices(sphere.Vertices.size());
	for (int i = 0; i < sphere.Vertices.size(); i++)
	{
		sphere_vertices[i].Pos = sphere.Vertices[i].Position;
		//sphere_vertices[i].Color = DirectX::XMFLOAT4(DirectX::Colors::Green);
		sphere_vertices[i].Normal = sphere.Vertices[i].Normal;
		sphere_vertices[i].TexC = sphere.Vertices[i].TexC;
	}
	std::vector<Geometry::Vertex> cylinder_vertices(cylinder.Vertices.size());
	for (int i = 0; i < cylinder.Vertices.size(); i++)
	{
		cylinder_vertices[i].Pos = cylinder.Vertices[i].Position;
		//cylinder_vertices[i].Color = DirectX::XMFLOAT4(DirectX::Colors::Blue);
		cylinder_vertices[i].Normal = cylinder.Vertices[i].Normal;
		cylinder_vertices[i].TexC = cylinder.Vertices[i].TexC;
	}

	MeshGeometryHelper helper(this);
	helper.PushSubmeshGeometry("sphere", sphere_vertices, sphere.GetIndices16());
	helper.PushSubmeshGeometry("cylinder", cylinder_vertices, cylinder.GetIndices16());
	RIManager.AddGeometry("pillar", helper.CreateMeshGeometry("pillar"));

	BuildRenderItem();
}

void QDirect3D12Widget::BuildRenderItem()
{
	UINT fllowObjCBIndex = 0;//����ȥ�ļ����峣��������CB�е�������2��ʼ
	//��Բ����Բ��ʵ��ģ�ʹ�����Ⱦ����
	for (int i = 0; i < 5; i++)
	{
		RenderItem* leftCylinderRitem = RIManager.AddRitem("pillar", "cylinder", RenderQueue::Opaque);
		RenderItem* rightCylinderRitem = RIManager.AddRitem("pillar", "cylinder", RenderQueue::Opaque);
		RenderItem* leftSphereRitem = RIManager.AddRitem("pillar", "sphere", RenderQueue::Opaque);
		RenderItem* rightSphereRitem = RIManager.AddRitem("pillar", "sphere", RenderQueue::Opaque);

		XMMATRIX leftCylWorld = XMMatrixTranslation(-5.0f, 1.5f, -10.0f + i * 5.0f);
		XMMATRIX rightCylWorld = XMMatrixTranslation(+5.0f, 1.5f, -10.0f + i * 5.0f);
		XMMATRIX leftSphereWorld = XMMatrixTranslation(-5.0f, 3.5f, -10.0f + i * 5.0f);
		XMMATRIX rightSphereWorld = XMMatrixTranslation(+5.0f, 3.5f, -10.0f + i * 5.0f);

		leftCylinderRitem->material = RIManager.mMaterials["grass"].get();
		rightCylinderRitem->material = RIManager.mMaterials["grass"].get();
		leftSphereRitem->material = RIManager.mMaterials["grass"].get();
		rightSphereRitem->material = RIManager.mMaterials["grass"].get();

		//���5��Բ��
		XMStoreFloat4x4(&(leftCylinderRitem->World), leftCylWorld);
		//�ұ�5��Բ��
		XMStoreFloat4x4(&(rightCylinderRitem->World), rightCylWorld);
		//���5����
		XMStoreFloat4x4(&(leftSphereRitem->World), leftSphereWorld);
		//�ұ�5����
		XMStoreFloat4x4(&(rightSphereRitem->World), rightSphereWorld);
	}
}
//void QDirect3D12Widget::BuildBoxGeometry()
//{
//	std::array<Geometry::Vertex, 8> vertices =
//	{
//		Geometry::Vertex({ XMFLOAT3(-1.0f, -1.0f, -1.0f),
//		XMFLOAT4(Colors::White) }),
//		Geometry::Vertex({ XMFLOAT3(-1.0f, +1.0f, -1.0f),
//		XMFLOAT4(Colors::Black) }),
//		Geometry::Vertex({ XMFLOAT3(+1.0f, +1.0f, -1.0f),
//		XMFLOAT4(Colors::Red) }),
//		Geometry::Vertex({ XMFLOAT3(+1.0f, -1.0f, -1.0f),
//		XMFLOAT4(Colors::Green) }),
//		Geometry::Vertex({ XMFLOAT3(-1.0f, -1.0f, +1.0f),
//		XMFLOAT4(Colors::Blue) }),
//		Geometry::Vertex({ XMFLOAT3(-1.0f, +1.0f, +1.0f),
//		XMFLOAT4(Colors::Yellow) }),
//		Geometry::Vertex({ XMFLOAT3(+1.0f, +1.0f, +1.0f),
//		XMFLOAT4(Colors::Cyan) }),
//		Geometry::Vertex({ XMFLOAT3(+1.0f, -1.0f, +1.0f),
//		XMFLOAT4(Colors::Magenta) })
//	};
//	std::array<std::uint16_t, 36> indices =
//	{
//		// front face
//		0, 1, 2,
//		0, 2, 3,
//		// back face
//		4, 6, 5,
//		4, 7, 6,
//		// left face
//		4, 5, 1,
//		4, 1, 0,
//		// right face
//		3, 2, 6,
//		3, 6, 7,
//		// top face
//		1, 5, 6,
//		1, 6, 2,
//		// bottom face
//		4, 0, 3,
//		4, 3, 7
//	};
//
//	const UINT vbByteSize = (UINT)vertices.size() * sizeof(Geometry::Vertex);
//	const UINT ibByteSize = (UINT)indices.size() * sizeof(std::uint16_t);
//
//	mMultiGeo = std::make_unique<Geometry::MeshGeometry>();
//	mMultiGeo->Name = "boxGeo";
//
//	ThrowIfFailed(D3DCreateBlob(vbByteSize, &mMultiGeo -> VertexBufferCPU));
//	CopyMemory(mMultiGeo->VertexBufferCPU -> GetBufferPointer(), vertices.data(), vbByteSize);
//	ThrowIfFailed(D3DCreateBlob(ibByteSize, &mMultiGeo -> IndexBufferCPU));
//	CopyMemory(mMultiGeo->IndexBufferCPU -> GetBufferPointer(), indices.data(), ibByteSize);
//
//	mMultiGeo->VertexBufferGPU = CreateDefaultBuffer(vbByteSize, vertices.data(), mMultiGeo->VertexBufferUploader);
//	mMultiGeo->IndexBufferGPU = CreateDefaultBuffer(ibByteSize, indices.data(), mMultiGeo->IndexBufferUploader);
//	mMultiGeo->VertexByteStride = sizeof(Geometry::Vertex);
//	mMultiGeo->VertexBufferByteSize = vbByteSize;
//	mMultiGeo->IndexFormat = DXGI_FORMAT_R16_UINT;
//	mMultiGeo->IndexBufferByteSize = ibByteSize;
//
//	Geometry::SubmeshGeometry submesh;
//	submesh.IndexCount = (UINT)indices.size();
//	submesh.StartIndexLocation = 0;
//	submesh.BaseVertexLocation = 0;
//	mMultiGeo->DrawArgs["box"] = submesh;
//}

void QDirect3D12Widget::BuildPSO()
{
	D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc;
	ZeroMemory(&psoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
	psoDesc.InputLayout = mpShader->GetInputLayout();
	psoDesc.pRootSignature = mRootSignature.Get();
	psoDesc.VS =
	{
		reinterpret_cast<BYTE*>(mpShader->vsBytecode->GetBufferPointer()),
			mpShader->vsBytecode->GetBufferSize()
	};
	psoDesc.PS =
	{
		reinterpret_cast<BYTE*>(mpShader->psBytecode->GetBufferPointer()),
			mpShader->psBytecode->GetBufferSize()
	};
	psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	psoDesc.SampleMask = UINT_MAX;
	psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	psoDesc.NumRenderTargets = 1;
	psoDesc.RTVFormats[0] = m_BackBufferFormat;
	psoDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;
	psoDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;
	psoDesc.DSVFormat = m_DepthStencilFormat;
	ThrowIfFailed(m_d3dDevice->CreateGraphicsPipelineState(&psoDesc,
		IID_PPV_ARGS(&mPSO)));
}

void QDirect3D12Widget::BuildFrameResources()
{
	for (int i = 0; i < frameResourcesCount; i++)
	{
		FrameResourcesArray.push_back(std::make_unique<FrameResource>(
			m_d3dDevice.Get(),
			1,     //passCount
			(UINT)RIManager.mAllRitems.size(),
			(UINT)RIManager.mMaterials.size(),
			vertex_num));	//objCount
	}
}

#pragma region QtSlot

/// <summary>
/// QT Slot :: On Frame
/// </summary>
void QDirect3D12Widget::onFrame()
{
	// Send ticked signal
	if (m_bRenderActive) emit ticked();

	m_tGameTimer.Tick();

	CalculateFrameState();
	Update();
	Draw();
}

/// <summary>
/// QT Slot :: On Resize
/// </summary>
void QDirect3D12Widget::onResize()
{
	// TODO(Gilad): FIXME: this needs to be done in a synchronized manner. Need to look at
	// DirectX-12 samples here: https://github.com/microsoft/DirectX-Graphics-Samples how to
	// properly do this without leaking memory.
	PauseFrames();
	// The window resized, so update the aspect ratio and recompute the
		// projection matrix.
	XMMATRIX P = XMMatrixPerspectiveFovLH(0.25f * MathHelper::Pi, (float)(width()) / (height()), 1.0f, 1000.0f);
	XMStoreFloat4x4(&mProj, P);

	//resizeSwapChain(width(), height());
	ContinueFrames();
}

#pragma endregion

#pragma region HelperFuncs
void QDirect3D12Widget::FlushCmdQueue()
{
	mCurrentFence++;	//CPU��������رպ󣬽���ǰΧ��ֵ+1
	m_CommandQueue->Signal(m_fence.Get(), mCurrentFence);	//��GPU������CPU���������󣬽�fence�ӿ��е�Χ��ֵ+1����fence->GetCompletedValue()+1
	if (m_fence->GetCompletedValue() < mCurrentFence)	//���С�ڣ�˵��GPUû�д�������������
	{
		HANDLE eventHandle = CreateEvent(nullptr, false, false, L"FenceSetDone");	//�����¼�
		m_fence->SetEventOnCompletion(mCurrentFence, eventHandle);//��Χ���ﵽmCurrentFenceֵ����ִ�е�Signal����ָ���޸���Χ��ֵ��ʱ������eventHandle�¼�
		WaitForSingleObject(eventHandle, INFINITE);//�ȴ�GPU����Χ���������¼���������ǰ�߳�ֱ���¼�������ע���Enent���������ٵȴ���
							   //���û��Set��Wait���������ˣ�Set��Զ������ã�����Ҳ��û�߳̿��Ի�������̣߳�
		CloseHandle(eventHandle);
	}
}
/// <summary>
/// 
/// </summary>
void QDirect3D12Widget::CalculateFrameState()
{
	static int frameCnt = 0;	//��֡��
	static float timeElapsed = 0.0f;	//���ŵ�ʱ��
	frameCnt++;	//ÿ֡++������һ����伴ΪFPSֵ
	//����ģ��
	/*std::wstring text = std::to_wstring(gt.TotalTime());
	std::wstring windowText = text;
	SetWindowText(mhMainWnd, windowText.c_str());*/
	//�ж�ģ��
	if (m_tGameTimer.GetTotalTime() - timeElapsed >= 1.0f)	//һ��>=0��˵���պù�һ��
	{
		float fps = (float)frameCnt;//ÿ�����֡
		float mspf = 1000.0f / fps;	//ÿ֡���ٺ���

		m_Fps = fps;
		m_TotalTime = m_tGameTimer.GetTotalTime();

		//Ϊ������һ��֡��ֵ������
		frameCnt = 0;
		timeElapsed += 1.0f;
	}
}


/// <summary>
/// A general way to create a ID3D12Resource
///  + reason: an intermediate upload buffer is required to initialize the data of a default buffer
/// </summary>
/// <param name="byteSize">		size of data	</param>
/// <param name="initData">		pointer to data </param>
/// <param name="uploadBuffer">	upload buffer	</param>
ComPtr<ID3D12Resource> QDirect3D12Widget::CreateDefaultBuffer
	(UINT64 byteSize, const void* initData, ComPtr<ID3D12Resource>& uploadBuffer)
{
	//����Ĭ�϶ѣ���Ϊ�ϴ��ѵ����ݴ������
	ComPtr<ID3D12Resource> defaultBuffer;

	// Create the actual default buffer resource.
	ThrowIfFailed(m_d3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),//����Ĭ�϶����͵Ķ�
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(byteSize),
		D3D12_RESOURCE_STATE_COMMON,//Ĭ�϶�Ϊ���մ洢���ݵĵط���������ʱ��ʼ��Ϊ��ͨ״̬
		nullptr,
		IID_PPV_ARGS(defaultBuffer.GetAddressOf())));

	// �����ϴ��ѣ������ǣ�д��CPU�ڴ����ݣ��������Ĭ�϶�
	// In order to copy CPU memory data into our default buffer, we need
	// to create an intermediate upload heap.
	ThrowIfFailed(m_d3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), //�����ϴ������͵Ķ�
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(byteSize),//����Ĺ��캯��������byteSize��������ΪĬ��ֵ������д
		D3D12_RESOURCE_STATE_GENERIC_READ,	//�ϴ��������Դ��Ҫ���Ƹ�Ĭ�϶ѣ������ǿɶ�״̬
		nullptr,	//�������ģ����Դ������ָ���Ż�ֵ
		IID_PPV_ARGS(uploadBuffer.GetAddressOf())));

	// Describe the data we want to copy into the default buffer.
	//�����ݴ�CPU�ڴ濽����GPU����
	D3D12_SUBRESOURCE_DATA subResourceData;
	subResourceData.pData = initData;
	subResourceData.RowPitch = byteSize;
	subResourceData.SlicePitch = subResourceData.RowPitch;

	//����Դ��COMMON״̬ת����COPY_DEST״̬��Ĭ�϶Ѵ�ʱ��Ϊ�������ݵ�Ŀ�꣩
	// Schedule to copy the data to the default buffer resource. 
	// At a high level, the helper function UpdateSubresources 
	// will copy the CPU memory into the intermediate upload heap. 
	// Then, using ID3D12CommandList::CopySubresourceRegion, 
	// the intermediate upload heap data will be copied to mBuffer.
	m_CommandList->ResourceBarrier(1,
		&CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(),
			D3D12_RESOURCE_STATE_COMMON,
			D3D12_RESOURCE_STATE_COPY_DEST));

	//���ĺ���UpdateSubresources�������ݴ�CPU�ڴ濽�����ϴ��ѣ��ٴ��ϴ��ѿ�����Ĭ�϶ѡ�1����������Դ���±꣨ģ���ж��壬��Ϊ��2������Դ��
	UpdateSubresources<1>(m_CommandList.Get(), defaultBuffer.Get(), uploadBuffer.Get(), 0, 0, 1, &subResourceData);

	//�ٴν���Դ��COPY_DEST״̬ת����GENERIC_READ״̬(����ֻ�ṩ����ɫ������)
	m_CommandList->ResourceBarrier(1,
		&CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(),
			D3D12_RESOURCE_STATE_COPY_DEST,
			D3D12_RESOURCE_STATE_GENERIC_READ));

	// Note: uploadBuffer has to be kept alive after the above function 
	// calls because the command list has not been executed yet that 
	// performs the actual copy. 
	// The caller can Release the uploadBuffer after it knows the copy 
	// has been executed.
	return defaultBuffer;
}

#pragma endregion

#pragma region InputCallback

void QDirect3D12Widget::OnMouseMove(QMouseEvent* event)
{
	MainCamera.OnMouseMove(event);
}

void QDirect3D12Widget::OnMousePressed(QMouseEvent* event)
{
	MainCamera.OnMousePressed(event);
}

void QDirect3D12Widget::OnMouseReleased(QMouseEvent* event)
{

}
#pragma endregion

#pragma region D3DInitialize
bool QDirect3D12Widget::InitDirect3D()
{
	// Enable the D3D12 debug layer.
#if defined(DEBUG) || defined(_DEBUG)
	{
		ComPtr<ID3D12Debug> debugController;
		ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)));
		debugController->EnableDebugLayer();
	}
#endif
	try
	{
		CreateDevice();
		CreateFence();
		GetDescriptorSize();
		SetMSAA();
		CreateCommandObjects();
		CreateSwapChain();
		CreateDescriptorHeap();
		CreateRTV();
		CreateDSV();
		CreateViewPortAndScissorRect();

		ThrowIfFailed(m_CommandList->Close());
		ID3D12CommandList* cmdLists[] = { m_CommandList.Get() };
		m_CommandQueue->ExecuteCommandLists(_countof(cmdLists), cmdLists);
	}
	catch (HrException& e)
	{
		MessageBox(nullptr, e.ToLPCWSTR(), L"HR  Failed", MB_OK);
		return 0;
	}

	return true;
}
/// <summary>
/// Initialize:: 1 Create the Device
/// </summary>
void QDirect3D12Widget::CreateDevice()
{
	DXCall(CreateDXGIFactory1(IID_PPV_ARGS(&m_dxgiFactory)));
	DXCall(D3D12CreateDevice(nullptr,   //�˲����������Ϊnullptr����ʹ����������
		D3D_FEATURE_LEVEL_12_0,         //Ӧ�ó�����ҪӲ����֧�ֵ���͹��ܼ���
		IID_PPV_ARGS(&m_d3dDevice)));    //���������豸
}
/// <summary>
/// Initialize:: 2 Create the Fance
/// </summary>
void QDirect3D12Widget::CreateFence()
{
	DXCall(m_d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
}
/// <summary>
/// Initialize:: 3 Create Descriptor Sizes
/// </summary>
void QDirect3D12Widget::GetDescriptorSize()
{
	m_rtvDescriptorSize = m_d3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	m_dsvDescriptorSize = m_d3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
	m_cbv_srv_uavDescriptorSize = m_d3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}
/// <summary>
/// Initialize:: 4 Check 4X MSAA Quality Support
/// </summary>
void QDirect3D12Widget::SetMSAA()
{
	msaaQualityLevels.Format = m_BackBufferFormat;	//UNORM�ǹ�һ��������޷�������
	msaaQualityLevels.SampleCount = 4;
	msaaQualityLevels.Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE;
	msaaQualityLevels.NumQualityLevels = 0;
	//��ǰͼ��������MSAA���ز�����֧�֣�ע�⣺�ڶ������������������������
	ThrowIfFailed(m_d3dDevice->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &msaaQualityLevels, sizeof(msaaQualityLevels)));
	//NumQualityLevels��Check��������������
	//���֧��MSAA����Check�������ص�NumQualityLevels > 0
	//expressionΪ�٣���Ϊ0��������ֹ�������У�����ӡһ��������Ϣ
	m4xMsaaQuality = msaaQualityLevels.NumQualityLevels;

	assert(msaaQualityLevels.NumQualityLevels > 0 && "Unexpected MSAA quality level.");
}
/// <summary>
/// Initialize:: 5 Create Command Queue and Command Lists
/// </summary>
void QDirect3D12Widget::CreateCommandObjects()
{
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

	DXCall(m_d3dDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_CommandQueue)));

	DXCall(m_d3dDevice->CreateCommandAllocator(
		D3D12_COMMAND_LIST_TYPE_DIRECT,
		IID_PPV_ARGS(m_DirectCmdListAlloc.GetAddressOf())));

	DXCall(m_d3dDevice->CreateCommandList(
		0, //����ֵΪ0����GPU
		D3D12_COMMAND_LIST_TYPE_DIRECT, //�����б�����
		m_DirectCmdListAlloc.Get(), // Associated command allocator	//����������ӿ�ָ��
		nullptr,                   // Initial PipelineStateObject	//��ˮ��״̬����PSO�����ﲻ���ƣ����Կ�ָ��
		IID_PPV_ARGS(m_CommandList.GetAddressOf())));	//���ش����������б�

	// Start off in a closed state.  This is because the first time we refer 
	// to the command list we will Reset it, and it needs to be closed before
	// calling Reset.
	m_CommandList->Close();	//���������б�ǰ���뽫��ر�

	m_CommandList->Reset(m_DirectCmdListAlloc.Get(), nullptr);
}
/// <summary>
/// Initialize:: 6 Describe and Create Swap Chain
/// </summary>
void QDirect3D12Widget::CreateSwapChain()
{
	// Release the previous swapchain we will be recreating.
	m_SwapChain.Reset();

	DXGI_SWAP_CHAIN_DESC swapChainDesc;	//�����������ṹ��
	swapChainDesc.BufferDesc.Width = width();	//�������ֱ��ʵĿ��
	swapChainDesc.BufferDesc.Height = height();	//�������ֱ��ʵĸ߶�
	swapChainDesc.BufferDesc.Format = m_BackBufferFormat;	//����������ʾ��ʽ
	swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;	//ˢ���ʵķ���
	swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;	//ˢ���ʵķ�ĸ
	swapChainDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;	//����ɨ��VS����ɨ��(δָ����)
	swapChainDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;	//ͼ�������Ļ�����죨δָ���ģ�
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;	//��������Ⱦ����̨������������Ϊ��ȾĿ�꣩
	swapChainDesc.OutputWindow = (HWND)winId();	//��Ⱦ���ھ��
	swapChainDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;	//���ز�������
	swapChainDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;	//���ز�������
	swapChainDesc.Windowed = true;	//�Ƿ񴰿ڻ�
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;	//�̶�д��
	swapChainDesc.BufferCount = 2;	//��̨������������˫���壩
	swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;	//����Ӧ����ģʽ���Զ�ѡ�������ڵ�ǰ���ڳߴ����ʾģʽ��
	//����DXGI�ӿ��µĹ����ഴ��������
	ThrowIfFailed(m_dxgiFactory->CreateSwapChain(m_CommandQueue.Get(), &swapChainDesc, m_SwapChain.GetAddressOf()));
}
/// <summary>
/// Initialize:: 7 Create the Descriptor Heaps
/// </summary>
void QDirect3D12Widget::CreateDescriptorHeap()
{
	//���ȴ���RTV��
	D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc;
	rtvDescriptorHeapDesc.NumDescriptors = 2;
	rtvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	rtvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvDescriptorHeapDesc.NodeMask = 0;
	ThrowIfFailed(m_d3dDevice->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));

	//Ȼ�󴴽�DSV��
	D3D12_DESCRIPTOR_HEAP_DESC dsvDescriptorHeapDesc;
	dsvDescriptorHeapDesc.NumDescriptors = 1;
	dsvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	dsvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	dsvDescriptorHeapDesc.NodeMask = 0;
	ThrowIfFailed(m_d3dDevice->CreateDescriptorHeap(&dsvDescriptorHeapDesc, IID_PPV_ARGS(&m_dsvHeap)));

	//Ȼ�󴴽�SRV��
	D3D12_DESCRIPTOR_HEAP_DESC srvDescriptorHeapDesc;
	srvDescriptorHeapDesc.NumDescriptors = 3;
	srvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	srvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	srvDescriptorHeapDesc.NodeMask = 0;
	ThrowIfFailed(m_d3dDevice->CreateDescriptorHeap(&srvDescriptorHeapDesc, IID_PPV_ARGS(&m_srvHeap)));
}
/// <summary>
/// Initialize:: 8 Create Render Target View
/// </summary>
void QDirect3D12Widget::CreateRTV()
{
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHeapHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart());
	for (int i = 0; i < 2; i++)
	{
		//��ô��ڽ������еĺ�̨��������Դ
		m_SwapChain->GetBuffer(i, IID_PPV_ARGS(m_SwapChainBuffer[i].GetAddressOf()));
		//����RTV
		m_d3dDevice->CreateRenderTargetView(m_SwapChainBuffer[i].Get(),
			nullptr,	//�ڽ������������Ѿ������˸���Դ�����ݸ�ʽ����������ָ��Ϊ��ָ��
			rtvHeapHandle);	//����������ṹ�壨�����Ǳ��壬�̳���CD3DX12_CPU_DESCRIPTOR_HANDLE��
		//ƫ�Ƶ����������е���һ��������
		rtvHeapHandle.Offset(1, m_rtvDescriptorSize);
	}
}
/// <summary>
/// Initialize:: 9 Create the Depth/Stencil Buffer & View
/// </summary>
void QDirect3D12Widget::CreateDSV()
{
	// Two steps to create a resource == buffer == ID3D12Resource
	// 
	// 1. Filling out a D3D12_RESOURCE_DESC structure 
	//	��CPU�д��������ģ��������Դ
	D3D12_RESOURCE_DESC dsvResourceDesc;
	dsvResourceDesc.Alignment = 0;	//ָ������
	dsvResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;	//ָ����Դά�ȣ����ͣ�ΪTEXTURE2D
	dsvResourceDesc.DepthOrArraySize = 1;	//�������Ϊ1
	dsvResourceDesc.Width = width();	//��Դ��
	dsvResourceDesc.Height = height();	//��Դ��
	dsvResourceDesc.MipLevels = 1;	//MIPMAP�㼶����
	dsvResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;	//ָ�������֣����ﲻָ����
	dsvResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;	//���ģ����Դ��Flag
	dsvResourceDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;	//24λ��ȣ�8λģ��,���и������͵ĸ�ʽDXGI_FORMAT_R24G8_TYPELESSҲ����ʹ��
	dsvResourceDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;	//���ز�������
	dsvResourceDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;	//���ز�������

	CD3DX12_CLEAR_VALUE optClear;	//�����Դ���Ż�ֵ��������������ִ���ٶȣ�CreateCommittedResource�����д��룩
	optClear.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;//24λ��ȣ�8λģ��,���и������͵ĸ�ʽDXGI_FORMAT_R24G8_TYPELESSҲ����ʹ��
	optClear.DepthStencil.Depth = 1;	//��ʼ���ֵΪ1
	optClear.DepthStencil.Stencil = 0;	//��ʼģ��ֵΪ0

	// 2. calling the ID3D12Device::CreateCommittedResource method
	//	����һ����Դ��һ���ѣ�������Դ�ύ�����У������ģ�������ύ��GPU�Դ��У�
	ThrowIfFailed(m_d3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),	//������ΪĬ�϶ѣ�����д�룩
		D3D12_HEAP_FLAG_NONE,	                //Flag
		&dsvResourceDesc,	                    //���涨���DSV��Դָ��
		D3D12_RESOURCE_STATE_COMMON,	        //��Դ��״̬Ϊ��ʼ״̬
		&optClear,	                            //���涨����Ż�ֵָ��
		IID_PPV_ARGS(&m_DepthStencilBuffer)));	//�������ģ����Դ

	m_d3dDevice->CreateDepthStencilView(m_DepthStencilBuffer.Get(),
		nullptr,	//D3D12_DEPTH_STENCIL_VIEW_DESC����ָ�룬����&dsvDesc
					//�����ڴ������ģ����Դʱ�Ѿ��������ģ���������ԣ������������ָ��Ϊ��ָ��
		m_dsvHeap->GetCPUDescriptorHandleForHeapStart());	//DSV���

	// Transition the resource from its initial state to be used as a depth buffer.
	m_CommandList->ResourceBarrier(1,	//Barrier���ϸ���
		&CD3DX12_RESOURCE_BARRIER::Transition(m_DepthStencilBuffer.Get(),
			D3D12_RESOURCE_STATE_COMMON,	//ת��ǰ״̬������ʱ��״̬����CreateCommittedResource�����ж����״̬��
			D3D12_RESOURCE_STATE_DEPTH_WRITE));
}

/// <summary>
/// Initialize:: 11 Create the Depth/Stencil Buffer & View
/// </summary>
void QDirect3D12Widget::CreateViewPortAndScissorRect()
{
	// Set viewport
	// �ӿ�����
	viewPort.TopLeftX = 0;
	viewPort.TopLeftY = 0;
	viewPort.Width = width();
	viewPort.Height = height();
	viewPort.MaxDepth = 1.0f;
	viewPort.MinDepth = 0.0f;
	// Set scissor rectangle
	// �ü��������ã�����������ض������޳���
	// ǰ����Ϊ���ϵ����꣬������Ϊ���µ�����
	scissorRect.left = 0;
	scissorRect.top = 0;
	scissorRect.right = width();
	scissorRect.bottom = height();
}
#pragma endregion

#pragma region QtOverrideEvent

/// <summary>
/// QT OVERRIDE EVENT :: Event
/// -------------------------------------
/// Deal with all kinds of input events
/// -------------------------------------
/// </summary>
bool QDirect3D12Widget::event(QEvent* event)
{
	switch (event->type())
	{
		// Workaround for https://bugreports.qt.io/browse/QTBUG-42183 to get key strokes.
		// To make sure that we always have focus on the widget when we enter the rect area.
	case QEvent::Enter:
	case QEvent::FocusIn:
	case QEvent::FocusAboutToChange:
		if (::GetFocus() != m_hWnd)
		{
			QWidget* nativeParent = this;
			while (true)
			{
				if (nativeParent->isWindow()) break;

				QWidget* parent = nativeParent->nativeParentWidget();
				if (!parent) break;

				nativeParent = parent;
			}

			if (nativeParent && nativeParent != this &&
				::GetFocus() == reinterpret_cast<HWND>(nativeParent->winId()))
				::SetFocus(m_hWnd);
		}
		break;
	case QEvent::KeyPress:
		if (!m_bRenderActive) break;
		if (((QKeyEvent*)event)->isAutoRepeat()) break;
		InputSys.KeyPressed((QKeyEvent*)event);
		//emit keyPressed((QKeyEvent*)event);
		break;
	case QEvent::KeyRelease:
		if (!m_bRenderActive) break;
		if (((QKeyEvent*)event)->isAutoRepeat()) break;
		InputSys.KeyReleased((QKeyEvent*)event);
		break;
	case QEvent::MouseMove:
		if (!m_bRenderActive) break;
		OnMouseMove((QMouseEvent*)event);
		emit mouseMoved((QMouseEvent*)event);
		break;
	case QEvent::MouseButtonPress:
		if (!m_bRenderActive) break;
		OnMousePressed((QMouseEvent*)event);
		emit mouseClicked((QMouseEvent*)event);
		break;
	case QEvent::MouseButtonRelease:
		if (!m_bRenderActive) break;
		OnMouseReleased((QMouseEvent*)event);
		emit mouseReleased((QMouseEvent*)event);
		break;
	}

	return QWidget::event(event);
}

/// <summary>
/// QT OVERRIDE EVENT :: ShowEvent
/// --------------------------------------------------------------------------
/// Actually kinds of initialization
/// Internal show events are delivered just before the widget becomes visible.
/// --------------------------------------------------------------------------
/// </summary>
void QDirect3D12Widget::showEvent(QShowEvent* event)
{
	if (!m_bDeviceInitialized)
	{
		m_bDeviceInitialized = Initialize();
		emit deviceInitialized(m_bDeviceInitialized);
	}

	QWidget::showEvent(event);
}

/// <summary>
/// QT OVERRIDE EVENT :: PaintEngine
/// -------------------------------------
/// provides an abstract definition of how 
/// QPainter draws to a given device on a given platform
/// -------------------------------------
/// </summary>
QPaintEngine* QDirect3D12Widget::paintEngine() const
{
	return Q_NULLPTR;
}

/// <summary>
/// QT OVERRIDE EVENT :: PaintEvent
/// -------------------------------------
/// -------------------------------------
/// </summary>
void QDirect3D12Widget::paintEvent(QPaintEvent* event) {}

/// <summary>
/// QT OVERRIDE EVENT :: ResizeEvent
/// -------------------------------------
/// Help with parent class to resize, 
/// also call onResize in the widget.
/// -------------------------------------
/// </summary>
void QDirect3D12Widget::resizeEvent(QResizeEvent* event)
{
	if (m_bDeviceInitialized)
	{
		//Debug Change
		onResize();
		emit widgetResized();
	}

	QWidget::resizeEvent(event);
}

/// <summary>
/// QT OVERRIDE EVENT :: WheelEvent
/// -------------------------------------
/// Deal with wheel input event
/// -------------------------------------
/// </summary>
void QDirect3D12Widget::wheelEvent(QWheelEvent* event)
{
	if (event->angleDelta().x() == 0)
	{
		// TODO: Update your camera position based on the delta value.
	}
	else if (event->angleDelta().x() !=
		0) // horizontal scrolling - mice with another side scroller.
	{
		// MouseWheelH += (float)(event->angleDelta().y() / WHEEL_DELTA);
	}
	else if (event->angleDelta().y() != 0)
	{
		// MouseWheel += (float)(event->angleDelta().y() / WHEEL_DELTA);
	}

	QWidget::wheelEvent(event);
}
#pragma endregion

#pragma region QtSystemFunc

LRESULT QDirect3D12Widget::WndProc(MSG* pMsg)
{
	// Process wheel events using Qt's event-system.
	if (pMsg->message == WM_MOUSEWHEEL || pMsg->message == WM_MOUSEHWHEEL) return false;

	return false;
}

#if QT_VERSION >= 0x050000
bool QDirect3D12Widget::nativeEvent(const QByteArray& eventType,
	void* message,
	long* result)
{
	Q_UNUSED(eventType);
	Q_UNUSED(result);

#    ifdef Q_OS_WIN
	MSG* pMsg = reinterpret_cast<MSG*>(message);
	return WndProc(pMsg);
#    endif

	return QWidget::nativeEvent(eventType, message, result);
}

#else // QT_VERSION < 0x050000
bool QDirect3D12Widget::winEvent(MSG* message, long* result)
{
	Q_UNUSED(result);

#    ifdef Q_OS_WIN
	MSG* pMsg = reinterpret_cast<MSG*>(message);
	return WndProc(pMsg);
#    endif

	return QWidget::winEvent(message, result);
}
#endif // QT_VERSION >= 0x050000

#pragma endregion
