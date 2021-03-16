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
	//如果GPU端围栏值小于CPU端围栏值，即CPU速度快于GPU，则令CPU等待
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
			//XMMATRIX赋值给XMFLOAT4X4
			XMStoreFloat4x4(&objConstants.world, XMMatrixTranspose(w));
			//将数据拷贝至GPU缓存
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
	//首先重置命令分配器cmdAllocator和命令列表cmdList，目的是重置命令和列表，复用相关内存。
	// Reuse the memory associated with command recording.
	// We can only reset when the associated command lists have finished
	// execution on the GPU.
	auto currCmdAllocator = mCurrFrameResource->cmdAllocator;
	DXCall(currCmdAllocator->Reset());//重复使用记录命令的相关内存
	DXCall(m_CommandList->Reset(currCmdAllocator.Get(), mPSO.Get()));//复用命令列表及其内存

	// Indicate a state transition on the resource usage.
	//接着我们将后台缓冲资源从呈现状态转换到渲染目标状态（即准备接收图像渲染）。
	UINT& ref_mCurrentBackBuffer = mCurrentBackBuffer;
	m_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_SwapChainBuffer[ref_mCurrentBackBuffer].Get(),//转换资源为后台缓冲区资源
		D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));//从呈现到渲染目标转换

	//接下来设置视口和裁剪矩形。
	m_CommandList->RSSetViewports(1, &viewPort);
	m_CommandList->RSSetScissorRects(1, &scissorRect);

	// Clear the back buffer and depth buffer.
	//然后清除后台缓冲区和深度缓冲区，并赋值。步骤是先获得堆中描述符句柄（即地址），再通过ClearRenderTargetView函数和ClearDepthStencilView函数做清除和赋值。这里我们将RT资源背景色赋值为DarkRed（暗红）。
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), ref_mCurrentBackBuffer, m_rtvDescriptorSize);
	m_CommandList->ClearRenderTargetView(rtvHandle, DirectX::Colors::DarkRed, 0, nullptr);//清除RT背景色为暗红，并且不设置裁剪矩形
	D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
	m_CommandList->ClearDepthStencilView(dsvHandle,	//DSV描述符句柄
		D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
		1.0f,	//默认深度值
		0,	//默认模板值
		0,	//裁剪矩形数量
		nullptr);	//裁剪矩形指针

	// Specify the buffers we are going to render to. 
	//mCommandList->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());
	//然后我们指定将要渲染的缓冲区，即指定RTV和DSV。
	m_CommandList->OMSetRenderTargets(1,//待绑定的RTV数量
		&rtvHandle,	//指向RTV数组的指针
		true,	//RTV对象在堆内存中是连续存放的
		&dsvHandle);	//指向DSV的指针

	//ID3D12DescriptorHeap* descriptorHeaps[] = { m_cbvHeap.Get() };
	//m_CommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
	m_CommandList->SetGraphicsRootSignature(mRootSignature.Get());
	//设置SRV描述符堆
	//注意这里之所以是数组，是因为还可能包含SRV和UAV，而这里我们只用到了SRV
	ID3D12DescriptorHeap* descriptorHeaps[] = { m_srvHeap.Get() };
	m_CommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

	UINT passConstSize = Utils::CalcConstantBufferByteSize(sizeof(PassConstants));
	auto passCB = mCurrFrameResource->passCB-> Resource();
	m_CommandList->SetGraphicsRootConstantBufferView(1, passCB-> GetGPUVirtualAddress());
	// Deprecated: use descriptor table
	//		int passCbvIndex = (int)mMultiGeo->RenderItems.size() * frameResourcesCount + currFrameResourcesIndex;
	//		auto handle = CD3DX12_GPU_DESCRIPTOR_HANDLE(m_cbvHeap->GetGPUDescriptorHandleForHeapStart());
	//		handle.Offset(passCbvIndex, m_cbv_srv_uavDescriptorSize);
	//		m_CommandList->SetGraphicsRootDescriptorTable(1, //根参数的起始索引 handle);

	DrawRenderItems();

	// Indicate a state transition on the resource usage.
	// 等到渲染完成，我们要将后台缓冲区的状态改成呈现状态，使其之后推到前台缓冲区显示。完了，关闭命令列表，等待传入命令队列。
	m_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_SwapChainBuffer[ref_mCurrentBackBuffer].Get(),
		D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));//从渲染目标到呈现
	// 完成命令的记录关闭命令列表 
	ThrowIfFailed(m_CommandList->Close());

	// Add the command list to the queue for execution.
	//等CPU将命令都准备好后，需要将待执行的命令列表加入GPU的命令队列。使用的是ExecuteCommandLists函数。
	ID3D12CommandList* commandLists[] = { m_CommandList.Get() };//声明并定义命令列表数组
	m_CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);//将命令从命令列表传至命令队列

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
	// Because we are on the GPU timeline, the new fence point won’t be
	// set until the GPU finishes processing all the commands prior to
	// this Signal().
	m_CommandQueue->Signal(m_fence.Get(), mCurrentFence);
	// Note that GPU could still be working on commands from previous
	// frames, but that is okay, because we are not touching any frame
	// resources associated with those frames.
}
void QDirect3D12Widget::DrawRenderItems()
{
	//将智能指针数组转换成普通指针数组
	std::vector<RenderItem*> ritems;
	for (auto& e : RIManager.mAllRitems)
		ritems.push_back(e.get());

	auto objectCB = mCurrFrameResource->objCB->Resource();
	INT objCBByteSize = Utils::CalcConstantBufferByteSize(sizeof(ObjectConstants));
	INT matCBByteSize = Utils::CalcConstantBufferByteSize(sizeof(MaterialConstants));

	//遍历渲染项数组
	for (size_t i = 0; i < ritems.size(); i++)
	{
		auto ritem = ritems[i];

		m_CommandList->IASetVertexBuffers(0, 1, &ritem->Geo->VertexBufferView());
		m_CommandList->IASetIndexBuffer(&ritem->Geo->IndexBufferView());
		m_CommandList->IASetPrimitiveTopology(ritem->PrimitiveType);

		CD3DX12_GPU_DESCRIPTOR_HANDLE tex( m_srvHeap -> GetGPUDescriptorHandleForHeapStart());
		tex.Offset(0, m_cbv_srv_uavDescriptorSize);
		m_CommandList->SetGraphicsRootDescriptorTable(3, tex);

		//设置根描述符,将根描述符与资源绑定
		auto objCB = mCurrFrameResource->objCB->Resource();
		auto objCBAddress = objCB->GetGPUVirtualAddress();
		objCBAddress += ritem->ObjCBIndex * objCBByteSize;
		m_CommandList->SetGraphicsRootConstantBufferView(0,//寄存器槽号
			objCBAddress);//子资源地址
	// Deprecated: use descriptor table
	//		UINT objCbvIndex = currFrameResourcesIndex * (UINT)mMultiGeo->RenderItems.size() + ritem->ObjCBIndex;
	//		auto handle = CD3DX12_GPU_DESCRIPTOR_HANDLE(m_cbvHeap->GetGPUDescriptorHandleForHeapStart());
	//		handle.Offset(objCbvIndex, m_cbv_srv_uavDescriptorSize);
	//		m_CommandList->SetGraphicsRootDescriptorTable(0, //根参数的起始索引
	//			handle);

		auto matCB = mCurrFrameResource->materialCB->Resource();
		D3D12_GPU_VIRTUAL_ADDRESS matCBAddress = matCB->GetGPUVirtualAddress();
		matCBAddress += ritem->material->MatCBIndex * matCBByteSize;
		m_CommandList->SetGraphicsRootConstantBufferView(2, matCBAddress);

		//绘制顶点（通过索引缓冲区绘制）
		m_CommandList->DrawIndexedInstanced(ritem->IndexCount, //每个实例要绘制的索引数
			1,	//实例化个数
			ritem->StartIndexLocation,	//起始索引位置
			ritem->BaseVertexLocation,	//子物体起始索引在全局索引中的位置
			0);	//实例化的高级技术，暂时设置为0
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
	UINT objectCount = (UINT)RIManager.mAllRitems.size();//物体总个数（包括实例）

	// 创建Descriptor heap, 并创建CBV
	// 首先创建cbv堆
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
	UINT objectCount = (UINT)RIManager.mAllRitems.size();//物体总个数（包括实例）
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
			objCB_Address += i * objCBByteSize;//子物体在常量缓冲区中的地址
			int heapIndex = objectCount * frameIndex + i;	//CBV堆中的CBV元素索引
			auto handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_cbvHeap->GetCPUDescriptorHandleForHeapStart());//获得CBV堆首地址
			handle.Offset(heapIndex, m_cbv_srv_uavDescriptorSize);	//CBV句柄（CBV堆中的CBV元素地址）
			//创建CBV描述符
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
		//创建CBV描述符
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
	//			创建由两个CBV所组成的描述符表
	CD3DX12_DESCRIPTOR_RANGE cbvTable0;
	cbvTable0.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);
	CD3DX12_DESCRIPTOR_RANGE cbvTable1;
	cbvTable1.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 1);
	// 描述符类型  - 描述符数量 - 描述符所绑定的寄存器槽号
	// ---- 1.3 add content to the root parameter
	slotRootParameter[0].InitAsDescriptorTable(1, &cbvTable0);
	slotRootParameter[1].InitAsDescriptorTable(1, &cbvTable1);

	// 2. Create Root Signature
	// -------------------------------------------------
	// A root signature is an array of root parameters.
	//根签名由一组根参数构成
	// ---- 2.1 fill out a structure CD3DX12_ROOT_SIGNATURE_DESC
	CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(2,
		slotRootParameter, 0, nullptr,
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
	// create a root signature with a single slot which points to a
	// descriptor range consisting of a single constant buffer
	// 用单个寄存器槽来创建一个根签名，该槽位指向一个仅含有单个常量缓冲区的描述符区域
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
	srvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV,	//描述符类型
		1,	//描述符表数量
		0);	//描述符所绑定的寄存器槽号

	auto staticSamplers = TextureHelper::GetStaticSamplers();	//获得静态采样器集合

	//根参数可以是描述符表、根描述符、根常量
	CD3DX12_ROOT_PARAMETER slotRootParameter[4];
	slotRootParameter[0].InitAsConstantBufferView(0);
	slotRootParameter[1].InitAsConstantBufferView(1);
	slotRootParameter[2].InitAsConstantBufferView(2);
	slotRootParameter[3].InitAsDescriptorTable(1,//Range数量
		&srvTable,	//Range指针
		D3D12_SHADER_VISIBILITY_PIXEL);	//该资源只能在像素着色器可读

	//根签名由一组根参数构成
	CD3DX12_ROOT_SIGNATURE_DESC rootSig(4, //根参数的数量
		slotRootParameter, //根参数指针
		staticSamplers.size(), 
		staticSamplers.data(),	//静态采样器指针
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
	//用单个寄存器槽来创建一个根签名，该槽位指向一个仅含有单个常量缓冲区的描述符区域
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
	UINT fllowObjCBIndex = 0;//接下去的几何体常量数据在CB中的索引从2开始
	//将圆柱和圆的实例模型存入渲染项中
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

		//左边5个圆柱
		XMStoreFloat4x4(&(leftCylinderRitem->World), leftCylWorld);
		//右边5个圆柱
		XMStoreFloat4x4(&(rightCylinderRitem->World), rightCylWorld);
		//左边5个球
		XMStoreFloat4x4(&(leftSphereRitem->World), leftSphereWorld);
		//右边5个球
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
	mCurrentFence++;	//CPU传完命令并关闭后，将当前围栏值+1
	m_CommandQueue->Signal(m_fence.Get(), mCurrentFence);	//当GPU处理完CPU传入的命令后，将fence接口中的围栏值+1，即fence->GetCompletedValue()+1
	if (m_fence->GetCompletedValue() < mCurrentFence)	//如果小于，说明GPU没有处理完所有命令
	{
		HANDLE eventHandle = CreateEvent(nullptr, false, false, L"FenceSetDone");	//创建事件
		m_fence->SetEventOnCompletion(mCurrentFence, eventHandle);//当围栏达到mCurrentFence值（即执行到Signal（）指令修改了围栏值）时触发的eventHandle事件
		WaitForSingleObject(eventHandle, INFINITE);//等待GPU命中围栏，激发事件（阻塞当前线程直到事件触发，注意此Enent需先设置再等待，
							   //如果没有Set就Wait，就死锁了，Set永远不会调用，所以也就没线程可以唤醒这个线程）
		CloseHandle(eventHandle);
	}
}
/// <summary>
/// 
/// </summary>
void QDirect3D12Widget::CalculateFrameState()
{
	static int frameCnt = 0;	//总帧数
	static float timeElapsed = 0.0f;	//流逝的时间
	frameCnt++;	//每帧++，经过一秒后其即为FPS值
	//调试模块
	/*std::wstring text = std::to_wstring(gt.TotalTime());
	std::wstring windowText = text;
	SetWindowText(mhMainWnd, windowText.c_str());*/
	//判断模块
	if (m_tGameTimer.GetTotalTime() - timeElapsed >= 1.0f)	//一旦>=0，说明刚好过一秒
	{
		float fps = (float)frameCnt;//每秒多少帧
		float mspf = 1000.0f / fps;	//每帧多少毫秒

		m_Fps = fps;
		m_TotalTime = m_tGameTimer.GetTotalTime();

		//为计算下一组帧数值而重置
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
	//创建默认堆，作为上传堆的数据传输对象
	ComPtr<ID3D12Resource> defaultBuffer;

	// Create the actual default buffer resource.
	ThrowIfFailed(m_d3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),//创建默认堆类型的堆
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(byteSize),
		D3D12_RESOURCE_STATE_COMMON,//默认堆为最终存储数据的地方，所以暂时初始化为普通状态
		nullptr,
		IID_PPV_ARGS(defaultBuffer.GetAddressOf())));

	// 创建上传堆，作用是：写入CPU内存数据，并传输给默认堆
	// In order to copy CPU memory data into our default buffer, we need
	// to create an intermediate upload heap.
	ThrowIfFailed(m_d3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), //创建上传堆类型的堆
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(byteSize),//变体的构造函数，传入byteSize，其他均为默认值，简化书写
		D3D12_RESOURCE_STATE_GENERIC_READ,	//上传堆里的资源需要复制给默认堆，所以是可读状态
		nullptr,	//不是深度模板资源，不用指定优化值
		IID_PPV_ARGS(uploadBuffer.GetAddressOf())));

	// Describe the data we want to copy into the default buffer.
	//将数据从CPU内存拷贝到GPU缓存
	D3D12_SUBRESOURCE_DATA subResourceData;
	subResourceData.pData = initData;
	subResourceData.RowPitch = byteSize;
	subResourceData.SlicePitch = subResourceData.RowPitch;

	//将资源从COMMON状态转换到COPY_DEST状态（默认堆此时作为接收数据的目标）
	// Schedule to copy the data to the default buffer resource. 
	// At a high level, the helper function UpdateSubresources 
	// will copy the CPU memory into the intermediate upload heap. 
	// Then, using ID3D12CommandList::CopySubresourceRegion, 
	// the intermediate upload heap data will be copied to mBuffer.
	m_CommandList->ResourceBarrier(1,
		&CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(),
			D3D12_RESOURCE_STATE_COMMON,
			D3D12_RESOURCE_STATE_COPY_DEST));

	//核心函数UpdateSubresources，将数据从CPU内存拷贝至上传堆，再从上传堆拷贝至默认堆。1是最大的子资源的下标（模板中定义，意为有2个子资源）
	UpdateSubresources<1>(m_CommandList.Get(), defaultBuffer.Get(), uploadBuffer.Get(), 0, 0, 1, &subResourceData);

	//再次将资源从COPY_DEST状态转换到GENERIC_READ状态(现在只提供给着色器访问)
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
	DXCall(D3D12CreateDevice(nullptr,   //此参数如果设置为nullptr，则使用主适配器
		D3D_FEATURE_LEVEL_12_0,         //应用程序需要硬件所支持的最低功能级别
		IID_PPV_ARGS(&m_d3dDevice)));    //返回所建设备
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
	msaaQualityLevels.Format = m_BackBufferFormat;	//UNORM是归一化处理的无符号整数
	msaaQualityLevels.SampleCount = 4;
	msaaQualityLevels.Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE;
	msaaQualityLevels.NumQualityLevels = 0;
	//当前图形驱动对MSAA多重采样的支持（注意：第二个参数即是输入又是输出）
	ThrowIfFailed(m_d3dDevice->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &msaaQualityLevels, sizeof(msaaQualityLevels)));
	//NumQualityLevels在Check函数里会进行设置
	//如果支持MSAA，则Check函数返回的NumQualityLevels > 0
	//expression为假（即为0），则终止程序运行，并打印一条出错信息
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
		0, //掩码值为0，单GPU
		D3D12_COMMAND_LIST_TYPE_DIRECT, //命令列表类型
		m_DirectCmdListAlloc.Get(), // Associated command allocator	//命令分配器接口指针
		nullptr,                   // Initial PipelineStateObject	//流水线状态对象PSO，这里不绘制，所以空指针
		IID_PPV_ARGS(m_CommandList.GetAddressOf())));	//返回创建的命令列表

	// Start off in a closed state.  This is because the first time we refer 
	// to the command list we will Reset it, and it needs to be closed before
	// calling Reset.
	m_CommandList->Close();	//重置命令列表前必须将其关闭

	m_CommandList->Reset(m_DirectCmdListAlloc.Get(), nullptr);
}
/// <summary>
/// Initialize:: 6 Describe and Create Swap Chain
/// </summary>
void QDirect3D12Widget::CreateSwapChain()
{
	// Release the previous swapchain we will be recreating.
	m_SwapChain.Reset();

	DXGI_SWAP_CHAIN_DESC swapChainDesc;	//交换链描述结构体
	swapChainDesc.BufferDesc.Width = width();	//缓冲区分辨率的宽度
	swapChainDesc.BufferDesc.Height = height();	//缓冲区分辨率的高度
	swapChainDesc.BufferDesc.Format = m_BackBufferFormat;	//缓冲区的显示格式
	swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;	//刷新率的分子
	swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;	//刷新率的分母
	swapChainDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;	//逐行扫描VS隔行扫描(未指定的)
	swapChainDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;	//图像相对屏幕的拉伸（未指定的）
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;	//将数据渲染至后台缓冲区（即作为渲染目标）
	swapChainDesc.OutputWindow = (HWND)winId();	//渲染窗口句柄
	swapChainDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;	//多重采样数量
	swapChainDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;	//多重采样质量
	swapChainDesc.Windowed = true;	//是否窗口化
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;	//固定写法
	swapChainDesc.BufferCount = 2;	//后台缓冲区数量（双缓冲）
	swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;	//自适应窗口模式（自动选择最适于当前窗口尺寸的显示模式）
	//利用DXGI接口下的工厂类创建交换链
	ThrowIfFailed(m_dxgiFactory->CreateSwapChain(m_CommandQueue.Get(), &swapChainDesc, m_SwapChain.GetAddressOf()));
}
/// <summary>
/// Initialize:: 7 Create the Descriptor Heaps
/// </summary>
void QDirect3D12Widget::CreateDescriptorHeap()
{
	//首先创建RTV堆
	D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc;
	rtvDescriptorHeapDesc.NumDescriptors = 2;
	rtvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	rtvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvDescriptorHeapDesc.NodeMask = 0;
	ThrowIfFailed(m_d3dDevice->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));

	//然后创建DSV堆
	D3D12_DESCRIPTOR_HEAP_DESC dsvDescriptorHeapDesc;
	dsvDescriptorHeapDesc.NumDescriptors = 1;
	dsvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	dsvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	dsvDescriptorHeapDesc.NodeMask = 0;
	ThrowIfFailed(m_d3dDevice->CreateDescriptorHeap(&dsvDescriptorHeapDesc, IID_PPV_ARGS(&m_dsvHeap)));

	//然后创建SRV堆
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
		//获得存于交换链中的后台缓冲区资源
		m_SwapChain->GetBuffer(i, IID_PPV_ARGS(m_SwapChainBuffer[i].GetAddressOf()));
		//创建RTV
		m_d3dDevice->CreateRenderTargetView(m_SwapChainBuffer[i].Get(),
			nullptr,	//在交换链创建中已经定义了该资源的数据格式，所以这里指定为空指针
			rtvHeapHandle);	//描述符句柄结构体（这里是变体，继承自CD3DX12_CPU_DESCRIPTOR_HANDLE）
		//偏移到描述符堆中的下一个缓冲区
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
	//	在CPU中创建好深度模板数据资源
	D3D12_RESOURCE_DESC dsvResourceDesc;
	dsvResourceDesc.Alignment = 0;	//指定对齐
	dsvResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;	//指定资源维度（类型）为TEXTURE2D
	dsvResourceDesc.DepthOrArraySize = 1;	//纹理深度为1
	dsvResourceDesc.Width = width();	//资源宽
	dsvResourceDesc.Height = height();	//资源高
	dsvResourceDesc.MipLevels = 1;	//MIPMAP层级数量
	dsvResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;	//指定纹理布局（这里不指定）
	dsvResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;	//深度模板资源的Flag
	dsvResourceDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;	//24位深度，8位模板,还有个无类型的格式DXGI_FORMAT_R24G8_TYPELESS也可以使用
	dsvResourceDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;	//多重采样数量
	dsvResourceDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;	//多重采样质量

	CD3DX12_CLEAR_VALUE optClear;	//清除资源的优化值，提高清除操作的执行速度（CreateCommittedResource函数中传入）
	optClear.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;//24位深度，8位模板,还有个无类型的格式DXGI_FORMAT_R24G8_TYPELESS也可以使用
	optClear.DepthStencil.Depth = 1;	//初始深度值为1
	optClear.DepthStencil.Stencil = 0;	//初始模板值为0

	// 2. calling the ID3D12Device::CreateCommittedResource method
	//	创建一个资源和一个堆，并将资源提交至堆中（将深度模板数据提交至GPU显存中）
	ThrowIfFailed(m_d3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),	//堆类型为默认堆（不能写入）
		D3D12_HEAP_FLAG_NONE,	                //Flag
		&dsvResourceDesc,	                    //上面定义的DSV资源指针
		D3D12_RESOURCE_STATE_COMMON,	        //资源的状态为初始状态
		&optClear,	                            //上面定义的优化值指针
		IID_PPV_ARGS(&m_DepthStencilBuffer)));	//返回深度模板资源

	m_d3dDevice->CreateDepthStencilView(m_DepthStencilBuffer.Get(),
		nullptr,	//D3D12_DEPTH_STENCIL_VIEW_DESC类型指针，可填&dsvDesc
					//由于在创建深度模板资源时已经定义深度模板数据属性，所以这里可以指定为空指针
		m_dsvHeap->GetCPUDescriptorHandleForHeapStart());	//DSV句柄

	// Transition the resource from its initial state to be used as a depth buffer.
	m_CommandList->ResourceBarrier(1,	//Barrier屏障个数
		&CD3DX12_RESOURCE_BARRIER::Transition(m_DepthStencilBuffer.Get(),
			D3D12_RESOURCE_STATE_COMMON,	//转换前状态（创建时的状态，即CreateCommittedResource函数中定义的状态）
			D3D12_RESOURCE_STATE_DEPTH_WRITE));
}

/// <summary>
/// Initialize:: 11 Create the Depth/Stencil Buffer & View
/// </summary>
void QDirect3D12Widget::CreateViewPortAndScissorRect()
{
	// Set viewport
	// 视口设置
	viewPort.TopLeftX = 0;
	viewPort.TopLeftY = 0;
	viewPort.Width = width();
	viewPort.Height = height();
	viewPort.MaxDepth = 1.0f;
	viewPort.MinDepth = 0.0f;
	// Set scissor rectangle
	// 裁剪矩形设置（矩形外的像素都将被剔除）
	// 前两个为左上点坐标，后两个为右下点坐标
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
