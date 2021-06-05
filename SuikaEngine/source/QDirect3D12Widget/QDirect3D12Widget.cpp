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
#include <assimp/anim.h>
#include <ModelLoader.h>

#include <CudaPrt.h>
#include <Scene.h>
#include <Options.h>


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
		std::cout << "Init finished" << std::endl;
		m_CommandList->Reset(m_DirectCmdListAlloc, nullptr);

		// Lights
		BuildLights();
		std::cout << "Build Light finished" << std::endl;

		m_MemoryManagerModule->RTDSSub.CreateWritableTexture("framebuffer", width(), height(), WritableTexture::WritableType::RenderTarget);
		m_MemoryManagerModule->RTDSSub.CreateWritableTexture("framebuffer", width(), height(), WritableTexture::WritableType::DepthStencil);

		std::cout << "Create Buffer finished" << std::endl;

		// CreateRTVDSVDescriptorHeap
		m_MemoryManagerModule->SetCommandList(m_CommandList);
		m_MemoryManagerModule->RTDSSub.CreateRTDSHeap(width(), height(), m_SwapChain.Get());

		std::cout << "Create RTDS Heap finished" << std::endl;

		// Shders
		BuildShadersAndInputLayout();
		std::cout << "Build shader finished" << std::endl;
		// Textures
		std::cout << "Read Textures Begin" << std::endl;
		BuildTexture();
		std::cout << "Read Textures Finished" << std::endl;
		// Materials
		BuildMaterial();
		std::cout << "Build Material Finished" << std::endl;
		// Geometry Things
		BuildGeometry();
		// Init Frame Resource,must after all render items pushed;
		// BuildFrameResources
		m_SynchronizationModule->BuildFrameResources(&RIManager, vertex_num);
		// Build Rootsignature
		BuildRootSignature();
		// Build PSOs
		BuildPSO();

		// Start the mission
		ThrowIfFailed(m_CommandList->Close());
		ID3D12CommandList* cmdLists[] = { m_CommandList };
		m_CommandQueue->ExecuteCommandLists(_countof(cmdLists), cmdLists);

		// Wait until initialization is complete.
		m_SynchronizationModule->SetMainCommandQueue(m_CommandQueue);
		m_SynchronizationModule->SynchronizeMainQueue();
		//m_SynchronizationModule->FlushCmdQueue(m_CommandQueue);
		m_CudaManagerModule->InitSynchronization(m_SynchronizationModule.get());
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
	m_SynchronizationModule->StartUpdate();
	mCurrFrameResource = m_SynchronizationModule->mCurrFrameResource;

	ObjectConstants& objConstants = m_ResourceBindingModule->GetObjectConstants();
	PassConstants& passConstants = m_ResourceBindingModule->GetPassConstants();

	MainCamera.Update();
	RIManager.mLights["mainLit"]->UpdateLightView();

	XMMATRIX view = MainCamera.GetViewMatrix();
	//XMStoreFloat4x4(&MainCamera.mView, view);
	XMMATRIX proj = XMLoadFloat4x4(&MainCamera.mProj);
	XMMATRIX viewProj = view * proj;
	// Update the constant buffer with the latest worldViewProj matrix.
	passConstants.gTime = m_tGameTimer.GetTotalTime();
	passConstants.eyePos = MainCamera.GetPosition();
	passConstants.light[0] = RIManager.mLights["mainLit"]->basic;
	passConstants.light[1] = RIManager.mLights["Light1"]->basic;
	passConstants.light[2] = RIManager.mLights["Light2"]->basic;
	passConstants.light[3] = RIManager.mLights["Light3"]->basic;
	XMStoreFloat4x4(&passConstants.viewProj, XMMatrixTranspose(viewProj));
	XMMATRIX gShadowTransform = XMLoadFloat4x4(&RIManager.mLights["mainLit"]->mShadowTransform);
	XMStoreFloat4x4(&passConstants.gShadowTransform, XMMatrixTranspose(gShadowTransform));
	m_ResourceBindingModule->UpdatePassConstants(mCurrFrameResource);

	// Update object Index
	for (auto& e : RIManager.mAllRitems)
	{
		if (e->NumFramesDirty > 0)
		{
			XMMATRIX w = XMLoadFloat4x4(&e->World);
			XMMATRIX t = XMLoadFloat4x4(&e->texTransform);
			//XMMATRIX赋值给XMFLOAT4X4
			XMStoreFloat4x4(&objConstants.world, XMMatrixTranspose(w));
			XMStoreFloat4x4(&objConstants.texTransform, XMMatrixTranspose(t));
			objConstants.materialIndex = e->material->MatCBIndex;
			//将数据拷贝至GPU缓存
			mCurrFrameResource->objCB->CopyData(e->ObjCBIndex, objConstants);

			e->NumFramesDirty--;
		}
	}

	// Update Wave
	auto currWavesVB = mCurrFrameResource->dynamicVB.get();
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
	RIManager.mGeometries["lakeGeo"]->VertexBufferGPU = currWavesVB->Resource();

	// Update Material
	auto currMatSB = mCurrFrameResource->materialSB.get();
	for (auto& e : RIManager.mMaterials)
	{
		Material* mat = e.second.get();//获得键值对的值，即Material指针（智能指针转普通指针）
		if (mat->NumFramesDirty > 0)
		{
			MaterialData matData;
			//将定义的材质属性传给常量结构体中的元素
			matData.diffuseAlbedo = mat->DiffuseAlbedo;
			matData.fresnelR0 = mat->FresnelR0;
			matData.roughness = mat->Roughness;
			matData.metalness = mat->Metalness;
			matData.emission = mat->Emission;
			XMMATRIX matTransform = XMLoadFloat4x4(&mat->MatTransform);
			XMStoreFloat4x4(&matData.matTransform, XMMatrixTranspose(matTransform));
			matData.diffuseMapIndex = mat->DiffuseSrvHeapIndex;//纹理在SRV堆中索引

			//将材质常量数据复制到常量缓冲区对应索引地址处
			currMatSB->CopyData(mat->MatCBIndex, matData);
			//更新下一个帧资源
			mat->NumFramesDirty--;
		}
	}
}

/// <summary>
/// LIFECYCLE :: Draw Stuff
/// </summary>
void QDirect3D12Widget::DrawShadowPass()
{
	RIManager.mLights["mainLit"]->UpdateShadowPassCB(mCurrFrameResource);	m_ResourceBindingModule->BindPassConstants(m_CommandList, mCurrFrameResource, 1);
	m_ResourceBindingModule->BindPassConstants(m_CommandList, mCurrFrameResource, 1);
	RIManager.mLights["mainLit"]->StartDrawShadowMap(m_CommandList, mCurrFrameResource, &RIManager);
	DrawRenderItems(RenderQueue::Opaque);
	RIManager.mLights["mainLit"]->EndDrawShadowMap(m_CommandList);
}

void QDirect3D12Widget::Draw()
{
	// 首先重置命令分配器cmdAllocator和命令列表cmdList，目的是重置命令和列表，复用相关内存。
	// Reuse the memory associated with command recording.
	// We can only reset when the associated command lists have finished
	// execution on the GPU.
	auto currCmdAllocator = mCurrFrameResource->cmdAllocator;

	// A command list can be reset after it has been added to the command queue via ExecuteCommandList.
	// Reusing the command list reuses memory.
	m_WorkSubmissionModule->ResetCommandList(m_CommandList, currCmdAllocator.Get(), RIManager.mPSOs["opaque"].Get());

	//Set SRV Descriptor Heap
	//注意这里之所以是数组，是因为还可能包含SRV和UAV，而这里我们只用到了SRV
	m_srvHeap = m_MemoryManagerModule->SRVSub.GetMainHeap();
	ID3D12DescriptorHeap* descriptorHeaps[] = { m_srvHeap };
	m_CommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

	//Set root signature
	m_CommandList->SetGraphicsRootSignature(mRootSignature.Get());

	// Bind all the materials used in this scene.  For structured buffers, we can bypass the heap and 
	// set as a root descriptor.
	//设置matSB的描述符(因为只绑定一次，所以不需要做地址偏移)
	auto matSB = mCurrFrameResource->materialSB->Resource();
	m_CommandList->SetGraphicsRootShaderResourceView(3,//根参数索引
		matSB->GetGPUVirtualAddress());//子资源地址

	// Bind all the textures used in this scene.  Observe
	// that we only have to specify the first descriptor in the table.  
	// The root signature knows how many descriptors are expected in the table.
	//设置描述符表，将纹理资源与流水线绑定
	CD3DX12_GPU_DESCRIPTOR_HANDLE tex(m_srvHeap->GetGPUDescriptorHandleForHeapStart());
	m_CommandList->SetGraphicsRootDescriptorTable(0, tex);

	//绑定CubeMap资源所对应的SRV描述符
	CD3DX12_GPU_DESCRIPTOR_HANDLE skyTexDescriptor(m_srvHeap->GetGPUDescriptorHandleForHeapStart());
	skyTexDescriptor.Offset(RIManager.mMaterials["sky"]->DiffuseSrvHeapIndex, m_cbv_srv_uavDescriptorSize);
	m_CommandList->SetGraphicsRootDescriptorTable(4, skyTexDescriptor);

	//绑定CubeMap资源所对应的SRV描述符
	CD3DX12_GPU_DESCRIPTOR_HANDLE cubeArrayTexDescriptor(m_srvHeap->GetGPUDescriptorHandleForHeapStart());
	cubeArrayTexDescriptor.Offset(RIManager.mTextures["cubearray"]->SrvIndex, m_cbv_srv_uavDescriptorSize);
	m_CommandList->SetGraphicsRootDescriptorTable(5, cubeArrayTexDescriptor);

	// Do the shadow pass
	DrawShadowPass();

	// Bind Constant Buffer Pass Constant
	UINT passConstSize = Utils::CalcConstantBufferByteSize(sizeof(PassConstants));
	auto passCB = mCurrFrameResource->passCB->Resource();
	m_CommandList->SetGraphicsRootConstantBufferView(2, passCB->GetGPUVirtualAddress());

	// ======================================================================================
	// RenderToTexture
	// ======================================================================================
	//static int i = 0;
	//static wchar_t* namebuf[6] = { L"1.bmp",L"2.bmp" ,L"3.bmp" ,L"4.bmp" ,L"5.bmp" ,L"6.bmp" };
	//if (i < 6)
	//{
	//	PassConstants& passConstants = m_ResourceBindingModule->GetPassConstants();

	//	XMMATRIX view = XMLoadFloat4x4(&MainCamera.mCubemapViews[i]);
	//	XMMATRIX proj = XMLoadFloat4x4(&MainCamera.mCubemapPerspective);
	//	XMMATRIX viewProj = view * proj;
	//	passConstants.eyePos = MainCamera.mCubemapPosition;
	//	XMStoreFloat4x4(&passConstants.viewProj, XMMatrixTranspose(viewProj));
	//	m_ResourceBindingModule->UpdatePassConstants(mCurrFrameResource, 2);
	//	m_ResourceBindingModule->BindPassConstants(m_CommandList, mCurrFrameResource, 2);

	//	// Draw
		m_MemoryManagerModule->RTDSSub.RenderToTexture("framebuffer", "framebuffer");
	//	m_CommandList->SetPipelineState(RIManager.mPSOs["Skybox"].Get());
	//	DrawRenderItems(RenderQueue::Skybox);

	//	m_ResourceBindingModule->BindPassConstants(m_CommandList, mCurrFrameResource, 0);
	//}
	// 
	//if (i < 1)
	//{
	//	m_MemoryManagerModule->RTDSSub.RenderToTexture("framebuffer", "framebuffer");
	//	m_CommandList->SetPipelineState(RIManager.mPSOs["Texture"].Get());
	//	DrawRenderItems(RenderQueue::PostProcessing);
	//	m_ResourceBindingModule->BindPassConstants(m_CommandList, mCurrFrameResource, 0);
	//}
		
	// ======================================================================================
	// RenderToScreen
	// ======================================================================================
	//m_MemoryManagerModule->RTDSSub.RenderToScreen();

	//分别设置PSO并绘制对应渲染项
	m_CommandList->SetPipelineState(RIManager.mPSOs["opaque"].Get());
	DrawRenderItems(RenderQueue::Opaque);

	//static bool have = false;
	//if (InputSys.KeyboardPressed[InputSystem::InputTypes::PrtScreen])
	//{
	//	if (!have)
	//	{
	//		Debug::Log("Print screen");
	//		m_MemoryManagerModule->RTDSSub.PrintWirtableTexture("shadowmap1", m_CommandQueue);
	//	}
	//	have = true;
	//}

	m_CommandList->SetPipelineState(RIManager.mPSOs["alphaTest"].Get());
	DrawRenderItems(RenderQueue::AlphaTest);

	m_CommandList->SetPipelineState(RIManager.mPSOs["transparent"].Get());
	DrawRenderItems(RenderQueue::Transparent);

	//绘制渲染项
	m_CommandList->SetPipelineState(RIManager.mPSOs["Skybox"].Get());
	DrawRenderItems(RenderQueue::Skybox);

	////绘制渲染项
	if (MainCamera.DoUseRT())
	{
		m_CudaManagerModule->MoveToNextFrame(&MainCamera);
		m_MemoryManagerModule->RTDSSub.RenderToScreen();
		m_CommandList->SetPipelineState(RIManager.mPSOs["Texture"].Get());
		DrawRenderItems(RenderQueue::PostProcessing);
	}
	else
	{
		D3DModules::ComputeInstance* horizontalBlur = m_ResourceBindingModule->ComputeSub.ComputeIns("HorizontalBlur");
		WritableTexture* framebuffer = m_MemoryManagerModule->RTDSSub.GetWrtiableTexture("framebuffer", WritableTexture::RenderTarget);
		horizontalBlur->Execute(m_CommandList, framebuffer);

		m_MemoryManagerModule->RTDSSub.RenderTextureToScreen("framebuffer");
		//m_MemoryManagerModule->RTDSSub.UnorderedAccessTextureToScreen("post1");
	}

	m_MemoryManagerModule->EndNewFrame();
	DXCall(m_CommandList->Close());

	// Add the command list to the queue for execution.
	//等CPU将命令都准备好后，需要将待执行的命令列表加入GPU的命令队列。使用的是ExecuteCommandLists函数。
	ID3D12CommandList* commandLists[] = { m_CommandList };//声明并定义命令列表数组
	m_CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);//将命令从命令列表传至命令队列
	//if (i < 6)
	//{
	//	m_MemoryManagerModule->RTDSSub.PrintWirtableTexture("framebuffer", namebuf[i], m_CommandQueue);
	//	i++;
	//}
	//if (i < 1)
	//{
	//	m_MemoryManagerModule->RTDSSub.PrintWirtableTexture("framebuffer", namebuf[i], m_CommandQueue);
	//	i++;
	//}
	// swap the back and front buffers
	ThrowIfFailed(m_SwapChain->Present(0, 0));

	m_SynchronizationModule->EndUpdate(m_CommandQueue);
}

void QDirect3D12Widget::DrawRenderItems(RenderQueue queue)
{
	//将智能指针数组转换成普通指针数组
	std::vector<RenderItem*> ritems;
	for (auto& e : (RIManager.mQueueRitems[queue]))
		ritems.push_back(e);

	auto objectCB = mCurrFrameResource->objCB->Resource();
	INT objCBByteSize = Utils::CalcConstantBufferByteSize(sizeof(ObjectConstants));

	//遍历渲染项数组
	for (size_t i = 0; i < ritems.size(); i++)
	{
		auto ritem = ritems[i];

		m_CommandList->IASetVertexBuffers(0, 1, &ritem->Geo->VertexBufferView());
		m_CommandList->IASetIndexBuffer(&ritem->Geo->IndexBufferView());
		m_CommandList->IASetPrimitiveTopology(ritem->PrimitiveType);

		//设置根描述符,将根描述符与资源绑定
		auto objCB = mCurrFrameResource->objCB->Resource();
		auto objCBAddress = objCB->GetGPUVirtualAddress();
		objCBAddress += ritem->ObjCBIndex * objCBByteSize;
		m_CommandList->SetGraphicsRootConstantBufferView(1,//寄存器槽号
			objCBAddress);//子资源地址

		// Deprecated: use descriptor table
		//		UINT objCbvIndex = currFrameResourcesIndex * (UINT)mMultiGeo->RenderItems.size() + ritem->ObjCBIndex;
		//		auto handle = CD3DX12_GPU_DESCRIPTOR_HANDLE(m_cbvHeap->GetGPUDescriptorHandleForHeapStart());
		//		handle.Offset(objCbvIndex, m_cbv_srv_uavDescriptorSize);
		//		m_CommandList->SetGraphicsRootDescriptorTable(0, //根参数的起始索引
		//			handle);

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
	, RIManager(this)
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

#pragma region Initialize

void QDirect3D12Widget::BuildShadersAndInputLayout()
{
	RIManager.mShaders["common"] = std::make_unique<Shader>(m_d3dDevice.Get(), L"Color");
	RIManager.mShaders["skybox"] = std::make_unique<Shader>(m_d3dDevice.Get(), L"Skybox");
	RIManager.mShaders["texture"] = std::make_unique<Shader>(m_d3dDevice.Get(), L"Texture");
	RIManager.mShaders["ShadowOpaque"] = std::make_unique<Shader>(m_d3dDevice.Get(), L"Shadow");
	m_ResourceBindingModule->ComputeSub.BuildPostProcessRootSignature();
	m_ResourceBindingModule->ComputeSub.CreateComputeInstance("HorizontalBlur", "PostProcessing", L"GaussianBlur", "HorzBlurCS");
	m_ResourceBindingModule->ComputeSub.CreateComputeInstance("VerticalBlur", "PostProcessing", L"GaussianBlur", "VerticalBlurCS");
}

void QDirect3D12Widget::BuildTexture()
{
	// Texture2D texture
	RIManager.PushTexture("lut", L"IBL_LUT.bmp");
	RIManager.PushTexture("wood", L"WoodCrate01.dds");
	RIManager.PushTexture("grass", L"grass.dds");
	RIManager.PushTexture("brick", L"bricks.dds");
	RIManager.PushTexture("water", L"water1.dds");
	RIManager.PushTexture("test", L"test.bmp");
	RIManager.PushTexture("ueno", L"IBL/FN_HDRI_035-gigapixel.png");
	RIManager.PushTexture("env", L"IBL/FN_HDRI_035.hdr");
	// Cuda texture
	RIManager.PushTextureCuda("cuda", (unsigned int)width(), (unsigned int)height());
	// Cubemap texture
	RIManager.PushTexture("cubeenv", L"Cubemaps/Desert/diffuse/.bmp", Texture::Type::Cubemap);
	RIManager.PushTexture("cubearray", L"Cubemaps/Desert/CubeArray.dds", Texture::Type::CubemapArray);
	// Create SRV
	WritableTexture* ua1 = m_MemoryManagerModule->RTDSSub.CreateWritableTexture("post1", width(), height(), WritableTexture::UnorderedAccess);
	WritableTexture* ua2 = m_MemoryManagerModule->RTDSSub.CreateWritableTexture("post2", width(), height(), WritableTexture::UnorderedAccess);
	//WritableTexture* cudatex = m_MemoryManagerModule->RTDSSub.CreateWritableTexture("cuda", width(), height(), WritableTexture::CudaShared);
	m_CudaManagerModule->SetTexture(RIManager.mTextures["cuda"].get());
	m_CudaManagerModule->BindTexture(RIManager.mTextures["env"].get());
	m_CudaManagerModule->SetEnvmap();
	D3DModules::ComputeInstance* horizontalBlur =  m_ResourceBindingModule->ComputeSub.ComputeIns("HorizontalBlur");
	horizontalBlur->ua1 = ua1;
	horizontalBlur->ua2 = ua2;
	m_MemoryManagerModule->SetSynchronizer(m_SynchronizationModule.get());
	m_MemoryManagerModule->InitSRVHeap(&RIManager);
}

void QDirect3D12Widget::BuildMaterial()
{
	std::unique_ptr<Material> grass = std::make_unique<Material>();
	grass->Name = "grass";
	grass->MatCBIndex = 0;
	grass->DiffuseAlbedo = XMFLOAT4(0.2f, 0.6f, 0.6f, 1.0f);
	grass->FresnelR0 = XMFLOAT3(0.01f, 0.01f, 0.01f);
	grass->Roughness = 0.125f;

	auto water = std::make_unique<Material>();
	water->Name = "water";
	water->MatCBIndex = 1;
	water->DiffuseAlbedo = XMFLOAT4(0.0f, 0.2f, 0.6f, 1.0f);
	water->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	water->Roughness = 0.0f;

	auto sky = std::make_unique<Material>();
	sky->Name = "water";
	sky->MatCBIndex = 2;
	sky->DiffuseAlbedo = XMFLOAT4(1.0f, 0.2f, 0.6f, 1.0f);
	sky->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	sky->Roughness = 0.0f;

	auto screen = std::make_unique<Material>();
	screen->Name = "screen";
	screen->MatCBIndex = 3;
	screen->DiffuseAlbedo = XMFLOAT4(0.0f, 0.2f, 0.6f, 1.0f);
	screen->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	screen->Roughness = 0.0f;

	auto standard = std::make_unique<Material>();
	standard->Name = "standard";
	standard->MatCBIndex = 4;
	standard->DiffuseAlbedo = XMFLOAT4(0.9f, 0.3f, 0.0f, 1.0f);
	standard->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	standard->Roughness = 0.2f;
	standard->Metalness = 0.4f;
	standard->Emission = XMFLOAT3(0, 0, 0);

	auto standard0 = std::make_unique<Material>();
	standard0->Name = "standard0";
	standard0->MatCBIndex = 5;
	standard0->DiffuseAlbedo = XMFLOAT4(0.1, 0.1, 1, 1);
	standard0->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	standard0->Roughness = 0.8f;
	standard0->Metalness = 0.3f;
	standard0->Emission = XMFLOAT3(0, 0, 0);

	auto standard1 = std::make_unique<Material>();
	standard1->Name = "standard1";
	standard1->MatCBIndex = 6;
	standard1->DiffuseAlbedo = XMFLOAT4(0.5f, 0.0f, 0.0f, 1.0f);
	standard1->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	standard1->Roughness = 1;
	standard1->Metalness = .9;
	standard1->Emission = XMFLOAT3(0, 0, 0);

	auto standard2 = std::make_unique<Material>();
	standard2->Name = "standard2";
	standard2->MatCBIndex = 7;
	standard2->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
	standard2->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	standard2->Roughness = 0;
	standard2->Metalness = 1;
	standard2->Emission = XMFLOAT3(0, 0, 0);

	auto standard3 = std::make_unique<Material>();
	standard3->Name = "standard3";
	standard3->MatCBIndex = 8;
	standard3->DiffuseAlbedo = XMFLOAT4(0.2f, 0.9f, 0.1f, 1.0f);
	standard3->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	standard3->Roughness = 0.2f;
	standard3->Metalness = 0.8f;
	standard3->Emission = XMFLOAT3(0, 0, 0);

	auto standard4 = std::make_unique<Material>();
	standard4->Name = "standard4";
	standard4->MatCBIndex = 9;
	standard4->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
	standard4->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	standard4->Roughness = 0.8f;
	standard4->Metalness = 0.3f;
	standard4->Emission = XMFLOAT3(1.f, 1.f, 1.f);

	//hitable[1] = new Sphere(make_float3(-0.5, -99.4, -1), 100, new DevLambertian(make_float3(0.5f, 0.0f, 0.0f), make_float3(0, 0, 0)));
	//hitable[2] = new Sphere(make_float3(1, 1, -1), 0.5, new DevMetal(make_float3(1, 1, 1), make_float3(0, 0, 0), 0));
	//hitable[3] = new Sphere(make_float3(-1, 1, -1), 0.5, new DevMetal(make_float3(0.2, 0.9, 0.1), make_float3(0, 0, 0), .4));
	//hitable[4] = new Sphere(make_float3(0, 3, 0), 1, new DevLambertian(make_float3(0.8, 0.8, 0.8), make_float3(6, 4, 2)));
	//hitable[5] = new TriangleModel(new DevCoat(make_float3(0.9f, 0.3f, 0.0f), make_float3(0, 0, 0), 0));
	RIManager.mMaterials["grass"] = std::move(grass);
	RIManager.mMaterials["water"] = std::move(water);
	RIManager.mMaterials["sky"] = std::move(sky);
	RIManager.mMaterials["screen"] = std::move(screen);
	RIManager.mMaterials["standard"] = std::move(standard);
	RIManager.mMaterials["standard0"] = std::move(standard0);
	RIManager.mMaterials["standard1"] = std::move(standard1);
	RIManager.mMaterials["standard2"] = std::move(standard2);
	RIManager.mMaterials["standard3"] = std::move(standard3);
	RIManager.mMaterials["standard4"] = std::move(standard4);

	RIManager.SetTexture("grass", "grass");
	RIManager.SetTexture("water", "water");
	RIManager.SetTexture("sky", "cubeenv");
	RIManager.SetTexture("screen", 9);
	RIManager.SetTexture("standard", -1);
}

void QDirect3D12Widget::BuildGeometry()
{
	BuildBoxGeometry();
	BuildMultiGeometry();
	BuildLandGeometry();
	BuildLakeGeometry();
	BuildScreenCanvasGeometry();
}

float GetHeight(float x, float z) {
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
	}
	std::vector<std::uint16_t> indices = grid.GetIndices16();

	MeshGeometryHelper helper(this);
	helper.PushSubmeshGeometry("grid", vertices, indices);
	helper.CalcNormal();
	RIManager.AddGeometry("landGeo", helper.CreateMeshGeometry("landGeo"));
	//RenderItem* land = RIManager.AddRitem("landGeo", "grid");
	////land->material = RIManager.mMaterials["grass"].get();
	//land->material = RIManager.mMaterials["standard"].get();
	//XMStoreFloat4x4(&land->texTransform, XMMatrixScaling(1.0f, 1.0f, 1.0f));
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
		vertices[i].Normal = XMFLOAT3(0, 0, 0);
		vertices[i].TexC = grid.Vertices[i].TexC;
	}
	std::vector<std::uint16_t> indices = grid.GetIndices16();

	MeshGeometryHelper helper(this);
	helper.PushSubmeshGeometry("grid", vertices, indices);
	RIManager.AddGeometry("lakeGeo", helper.CreateMeshGeometry("lakeGeo"));
	//RenderItem* lake = RIManager.AddRitem("lakeGeo", "grid", RenderQueue::Transparent);
	////lake->material = RIManager.mMaterials["water"].get();
	//lake->material = RIManager.mMaterials["standard"].get();
	//XMStoreFloat4x4(&lake->texTransform, XMMatrixScaling(1.0f, 1.0f, 1.0f));

	wave = std::make_unique<Waves>(std::move(helper), vertices.size());

	vertex_num = vertices.size();
}

void QDirect3D12Widget::BuildScreenCanvasGeometry()
{
	ProceduralGeometry::GeometryGenerator geoGen;
	ProceduralGeometry::GeometryGenerator::MeshData screen = geoGen.CreateScreenQuad();

	std::vector<Geometry::Vertex> box_vertices(screen.Vertices.size());
	for (int i = 0; i < screen.Vertices.size(); i++)
	{
		box_vertices[i].Pos = screen.Vertices[i].Position;
		box_vertices[i].Normal = screen.Vertices[i].Normal;
		box_vertices[i].TexC = screen.Vertices[i].TexC;
	}

	MeshGeometryHelper helper(this);
	helper.PushSubmeshGeometry("screen", box_vertices, screen.GetIndices16());
	RIManager.AddGeometry("screen", helper.CreateMeshGeometry("screen"));
	RenderItem* screenRitem = RIManager.AddRitem("screen", "screen", RenderQueue::PostProcessing);
	screenRitem->material = RIManager.mMaterials["screen"].get();
}

void QDirect3D12Widget::BuildLights()
{
	std::unique_ptr<Light> light = std::make_unique<Light>(true);
	light->basic.Direction = XMFLOAT3(.7, -.4, .7);
	light->basic.Position = XMFLOAT3(0, 1, 0);
	light->basic.Strength = XMFLOAT3(1, 1, 1);
	WritableTexture* shadowmap1 = m_MemoryManagerModule->RTDSSub.CreateWritableTexture("shadowmap1", 1024, 1024, WritableTexture::WritableType::DepthStencil);
	light->SetShadowMap(shadowmap1);
	RIManager.AddLight("mainLit", light);

	std::unique_ptr<Light> light1 = std::make_unique<Light>();
	light1->basic.Direction = XMFLOAT3(0, -1, 0);
	light1->basic.Position = XMFLOAT3(2, 1, 0);
	light1->basic.Strength = XMFLOAT3(1, 1, 1);
	RIManager.AddLight("Light1", light1);

	std::unique_ptr<Light> light2 = std::make_unique<Light>();
	light2->basic.Direction = XMFLOAT3(0, -1, 0);
	light2->basic.Position = XMFLOAT3(0, 1, 3);
	light2->basic.Strength = XMFLOAT3(1, 1, 1);
	RIManager.AddLight("Light2", light2);

	std::unique_ptr<Light> light3 = std::make_unique<Light>();
	light3->basic.Direction = XMFLOAT3(0, -1, 0);
	light3->basic.Position = XMFLOAT3(2, 1, -3);
	light3->basic.Strength = XMFLOAT3(1, 1, 1);
	RIManager.AddLight("Light3", light3);
}

#pragma endregion

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
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	//使用描述符表
	//绑定CubeMap的Range
	CD3DX12_DESCRIPTOR_RANGE srvTableCube;
	srvTableCube.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV,	//描述符类型
		1,	//表中的描述符数量（纹理数量）
		0);	//描述符所绑定的寄存器槽号

	CD3DX12_DESCRIPTOR_RANGE srvTableCubeArray;
	srvTableCubeArray.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV,	//描述符类型
		1,	//表中的描述符数量（纹理数量）
		1);	//描述符所绑定的寄存器槽号

	CD3DX12_DESCRIPTOR_RANGE srvTable;
	srvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV,	//描述符类型
		RIManager.mTextures.size(),	//描述符表数量
		2);	//描述符所绑定的寄存器槽号

	// Root parameter can be a table, root descriptor or root constants.
	//根参数可以是描述符表、根描述符、根常量
	CD3DX12_ROOT_PARAMETER slotRootParameter[6];
	slotRootParameter[1].InitAsConstantBufferView(0);
	slotRootParameter[2].InitAsConstantBufferView(1);
	//matSB绑定槽号为0的寄存器（和纹理公用一个SRV寄存器，但是不同Space）
	//StructureBuffer必须使用SRV或者UAV来绑定
	slotRootParameter[3].InitAsShaderResourceView(/*寄存器槽号*/0, /*RegisterSpace*/ 1);
	// Perfomance TIP: Order from most frequent to least frequent.
	slotRootParameter[0].InitAsDescriptorTable(1,//Range数量
		&srvTable,	//Range指针
		D3D12_SHADER_VISIBILITY_ALL);	//该资源只能在像素着色器可读
	slotRootParameter[4].InitAsDescriptorTable(1,//Range数量
		&srvTableCube,	//Range指针
		D3D12_SHADER_VISIBILITY_PIXEL);	//该资源只能在像素着色器可读
	slotRootParameter[5].InitAsDescriptorTable(1,//Range数量
		&srvTableCubeArray,	//Range指针
		D3D12_SHADER_VISIBILITY_PIXEL);	//该资源只能在像素着色器可读

	auto staticSamplers = TextureHelper::GetStaticSamplers();	//获得静态采样器集合
	//根签名由一组根参数构成
	CD3DX12_ROOT_SIGNATURE_DESC rootSig(6, //根参数的数量
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

void QDirect3D12Widget::BuildBoxGeometry()
{
	ProceduralGeometry::GeometryGenerator geoGen;
	ProceduralGeometry::GeometryGenerator::MeshData box = geoGen.CreateCube();

	// PRT need
	std::vector<Geometry::Vertex> box_vertices(box.Vertices.size());
	for (int i = 0; i < box.Vertices.size(); i++)
	{
		box_vertices[i].Pos = box.Vertices[i].Position;
		box_vertices[i].Normal = box.Vertices[i].Normal;
		box_vertices[i].TexC = box.Vertices[i].TexC;
	}

	MeshGeometryHelper helper(this);
	helper.PushSubmeshGeometry("cube", box_vertices, box.GetIndices16());
	RIManager.AddGeometry("cube", helper.CreateMeshGeometry("cube"));
	RenderItem* skyboxRitem = RIManager.AddRitem("cube", "cube", RenderQueue::Skybox);
	skyboxRitem->material = RIManager.mMaterials["sky"].get();


	ProceduralGeometry::GeometryGenerator::MeshData grid = geoGen.CreateGrid(100.0f, 100.0f, 2, 2);
	std::vector<Geometry::Vertex> grid_vertices(grid.Vertices.size());
	for (int i = 0; i < grid.Vertices.size(); i++)
	{
		grid_vertices[i].Pos = grid.Vertices[i].Position;
		grid_vertices[i].Normal = grid.Vertices[i].Normal;
		grid_vertices[i].TexC = grid.Vertices[i].TexC;
	}

	ModelLoader loader_dragon(this, true);
	loader_dragon.Load("../Resources/Models/dragon_plane.fbx");
	Suika::CudaTriangleModel* model = loader_dragon.GetCudaTriangle();
	scene = new Suika::Scene();
	scene->models.push_back(model);
	m_CudaManagerModule->CreateScene(*scene);

	// Step3: Load Transfer and Finish the Geometry Buffer
	if (!loader_dragon.LoadTransfer())
	{
		// If failed
		unsigned size = loader_dragon.helper.interm.g_verticesNo;
		CuVertex* vertices = loader_dragon.helper.interm.g_vertices;
		PRTransfer dragonPrt(vertices, size);
		RunPrecomputeTransfer(&dragonPrt);
		dragonPrt.WriteToFile(TRANSFER_PATH + loader_dragon.name + ".transfer");
		loader_dragon.LoadTransfer();
	}
	RIManager.AddGeometry("dragon", loader_dragon.GetMeshGeometry());

	for (int i = 0; i < loader_dragon.subnum; i++)
	{
		RenderItem* dragonItem = RIManager.AddRitem("dragon", loader_dragon.subname[i], RenderQueue::Opaque);
		dragonItem->material = RIManager.mMaterials["standard"].get();
		XMStoreFloat4x4(&dragonItem->World, XMMatrixScaling(10, 10, 10));
	}

	// NO PRT
	ModelLoader loader_light(this, true);
	loader_light.Load("../Resources/Models/test.obj");
	//RIManager.AddGeometry("light", loader_light.GetMeshGeometry());
	//for (int i = 0; i < loader_light.subnum; i++)
	//{
	//	RenderItem* lightRitem = RIManager.AddRitem("light", loader_light.subname[i], RenderQueue::Opaque);
	//	lightRitem->material = RIManager.mMaterials["standard"].get();
	//	XMStoreFloat4x4(&lightRitem->World, XMMatrixScaling(10, 10, 10));
	//}
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
		sphere_vertices[i].Normal = sphere.Vertices[i].Normal;
		sphere_vertices[i].TexC = sphere.Vertices[i].TexC;
	}
	//std::vector<Geometry::Vertex> cylinder_vertices(cylinder.Vertices.size());
	//for (int i = 0; i < cylinder.Vertices.size(); i++)
	//{
	//	cylinder_vertices[i].Pos = cylinder.Vertices[i].Position;
	//	cylinder_vertices[i].Normal = cylinder.Vertices[i].Normal;
	//	cylinder_vertices[i].TexC = cylinder.Vertices[i].TexC;
	//}

	MeshGeometryHelper helper(this);
	helper.PushSubmeshGeometry("sphere", sphere_vertices, sphere.GetIndices16());
	//helper.PushSubmeshGeometry("cylinder", cylinder_vertices, cylinder.GetIndices16());
	//RIManager.AddGeometry("pillar", helper.CreateMeshGeometry("pillar"));

	////将圆柱和圆的实例模型存入渲染项中
	//for (int i = 0; i < 5; i++)
	//{
	//	RenderItem* leftCylinderRitem = RIManager.AddRitem("pillar", "cylinder", RenderQueue::Opaque);
	//	RenderItem* rightCylinderRitem = RIManager.AddRitem("pillar", "cylinder", RenderQueue::Opaque);
	//	RenderItem* leftSphereRitem = RIManager.AddRitem("pillar", "sphere", RenderQueue::Opaque);
	//	RenderItem* rightSphereRitem = RIManager.AddRitem("pillar", "sphere", RenderQueue::Opaque);

	//	XMMATRIX leftCylWorld = XMMatrixTranslation(-5.0f, 1.5f, -10.0f + i * 5.0f);
	//	XMMATRIX rightCylWorld = XMMatrixTranslation(+5.0f, 1.5f, -10.0f + i * 5.0f);
	//	XMMATRIX leftSphereWorld = XMMatrixTranslation(-5.0f, 3.5f, -10.0f + i * 5.0f);
	//	XMMATRIX rightSphereWorld = XMMatrixTranslation(+5.0f, 3.5f, -10.0f + i * 5.0f);

	//	//leftCylinderRitem->material = RIManager.mMaterials["grass"].get();
	//	//rightCylinderRitem->material = RIManager.mMaterials["grass"].get();
	//	//leftSphereRitem->material = RIManager.mMaterials["grass"].get();
	//	//rightSphereRitem->material = RIManager.mMaterials["grass"].get();

	//	leftCylinderRitem->material = RIManager.mMaterials["standard"].get();
	//	rightCylinderRitem->material = RIManager.mMaterials["standard"].get();
	//	leftSphereRitem->material = RIManager.mMaterials["standard"].get();
	//	rightSphereRitem->material = RIManager.mMaterials["standard"].get();

	//	//左边5个圆柱
	//	XMStoreFloat4x4(&(leftCylinderRitem->World), leftCylWorld);
	//	//右边5个圆柱
	//	XMStoreFloat4x4(&(rightCylinderRitem->World), rightCylWorld);
	//	//左边5个球
	//	XMStoreFloat4x4(&(leftSphereRitem->World), leftSphereWorld);
	//	//右边5个球
	//	XMStoreFloat4x4(&(rightSphereRitem->World), rightSphereWorld);
	//}

	//MeshGeometryHelper helper_sphere(this);
	//helper_sphere.PushSubmeshGeometry("sphere", sphere_vertices, sphere.GetIndices16());

	//RIManager.AddGeometry("sphere", helper.CreateMeshGeometry("sphere"));
	//RenderItem* sphere_ritem = RIManager.AddRitem("sphere", "sphere", RenderQueue::Opaque);
	//XMMATRIX sphere_world = XMMatrixTranslation(0, 1, 1.2);
	//sphere_ritem->material = RIManager.mMaterials["standard0"].get();
	//XMStoreFloat4x4(&(sphere_ritem->World), sphere_world);

	//RenderItem* sphere_ritem2 = RIManager.AddRitem("sphere", "sphere", RenderQueue::Opaque);
	//sphere_ritem2->material = RIManager.mMaterials["standard1"].get();
	//XMMATRIX sphere_world2 = XMMatrixScaling(200, 200, 200) * XMMatrixTranslation(-0.5, -99.4, -1);
	//XMStoreFloat4x4(&(sphere_ritem2->World), sphere_world2);

	//RenderItem* sphere_ritem3 = RIManager.AddRitem("sphere", "sphere", RenderQueue::Opaque);
	//sphere_ritem3->material = RIManager.mMaterials["standard2"].get();
	//XMMATRIX sphere_world3 = XMMatrixTranslation(1, 1, -1);
	//XMStoreFloat4x4(&(sphere_ritem3->World), sphere_world3);

	//RenderItem* sphere_ritem4 = RIManager.AddRitem("sphere", "sphere", RenderQueue::Opaque);
	//sphere_ritem4->material = RIManager.mMaterials["standard3"].get();
	//XMMATRIX sphere_world4 = XMMatrixTranslation(-1, 1, -1);
	//XMStoreFloat4x4(&(sphere_ritem4->World), sphere_world4);

	//RenderItem* sphere_ritem5 = RIManager.AddRitem("sphere", "sphere", RenderQueue::Opaque);
	//sphere_ritem5->material = RIManager.mMaterials["standard4"].get();
	//XMMATRIX sphere_world5 = XMMatrixScaling(2, 2, 2) * XMMatrixTranslation(0, 3, 0);
	//XMStoreFloat4x4(&(sphere_ritem5->World), sphere_world5);
}

void QDirect3D12Widget::BuildPSO()
{
	D3D12_GRAPHICS_PIPELINE_STATE_DESC opaquePsoDesc;
	ZeroMemory(&opaquePsoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
	opaquePsoDesc.InputLayout = RIManager.mShaders["common"]->GetInputLayout();
	opaquePsoDesc.pRootSignature = mRootSignature.Get();
	opaquePsoDesc.VS =
	{
		reinterpret_cast<BYTE*>(RIManager.mShaders["common"]->vsBytecode->GetBufferPointer()),
			RIManager.mShaders["common"]->vsBytecode->GetBufferSize()
	};
	opaquePsoDesc.PS =
	{
		reinterpret_cast<BYTE*>(RIManager.mShaders["common"]->psBytecode->GetBufferPointer()),
			RIManager.mShaders["common"]->psBytecode->GetBufferSize()
	};
	opaquePsoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	opaquePsoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	opaquePsoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	opaquePsoDesc.SampleMask = UINT_MAX;
	opaquePsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	opaquePsoDesc.NumRenderTargets = 1;
	opaquePsoDesc.RTVFormats[0] = m_BackBufferFormat;
	opaquePsoDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;
	opaquePsoDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;
	opaquePsoDesc.DSVFormat = m_DepthStencilFormat;
	ThrowIfFailed(m_d3dDevice->CreateGraphicsPipelineState(&opaquePsoDesc,
		IID_PPV_ARGS(&RIManager.mPSOs["opaque"])));

	//AlphaTest物体的PSO（不需要混合）
	D3D12_GRAPHICS_PIPELINE_STATE_DESC alphaTestPsoDesc = opaquePsoDesc;//使用不透明物体的PSO初始化
	alphaTestPsoDesc.PS =
	{
		reinterpret_cast<BYTE*>(RIManager.mShaders["common"]->psBytecodeAlphaTest->GetBufferPointer()),
		RIManager.mShaders["common"]->psBytecodeAlphaTest->GetBufferSize()
	};
	alphaTestPsoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;//双面显示
	ThrowIfFailed(m_d3dDevice->CreateGraphicsPipelineState(&opaquePsoDesc,
		IID_PPV_ARGS(&RIManager.mPSOs["alphaTest"])));


	//混合物体的PSO（需要混合）
	D3D12_GRAPHICS_PIPELINE_STATE_DESC transparentPsoDesc = opaquePsoDesc;//使用不透明物体的PSO初始化
	D3D12_RENDER_TARGET_BLEND_DESC transparencyBlendDesc;
	transparencyBlendDesc.BlendEnable = true;	//是否开启常规混合（默认值为false）
	transparencyBlendDesc.LogicOpEnable = false;	//是否开启逻辑混合(默认值为false)
	transparencyBlendDesc.SrcBlend = D3D12_BLEND_SRC_ALPHA;	//RGB混合中的源混合因子Fsrc（这里取源颜色的alpha通道值）
	transparencyBlendDesc.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;//RGB混合中的目标混合因子Fdest（这里取1-alpha）
	transparencyBlendDesc.BlendOp = D3D12_BLEND_OP_ADD;	//RGB混合运算符(这里选择加法)
	transparencyBlendDesc.SrcBlendAlpha = D3D12_BLEND_ONE;	//alpha混合中的源混合因子Fsrc（取1）
	transparencyBlendDesc.DestBlendAlpha = D3D12_BLEND_ZERO;//alpha混合中的目标混合因子Fsrc（取0）
	transparencyBlendDesc.BlendOpAlpha = D3D12_BLEND_OP_ADD;//alpha混合运算符(这里选择加法)
	transparencyBlendDesc.LogicOp = D3D12_LOGIC_OP_NOOP;	//逻辑混合运算符(空操作，即不使用)
	transparencyBlendDesc.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;//后台缓冲区写入遮罩（没有遮罩，即全部写入）

	transparentPsoDesc.BlendState.RenderTarget[0] = transparencyBlendDesc;//赋值RenderTarget第一个元素，即对每一个渲染目标执行相同操作

	ThrowIfFailed(m_d3dDevice->CreateGraphicsPipelineState(&transparentPsoDesc,
		IID_PPV_ARGS(&RIManager.mPSOs["transparent"])));

	// 天空盒的PSO
	D3D12_GRAPHICS_PIPELINE_STATE_DESC skySpherePsoDesc = opaquePsoDesc;
	skySpherePsoDesc.VS =
	{
		reinterpret_cast<BYTE*>(RIManager.mShaders["skybox"]->vsBytecode->GetBufferPointer()),
		RIManager.mShaders["skybox"]->vsBytecode->GetBufferSize()
	};
	skySpherePsoDesc.PS =
	{
		reinterpret_cast<BYTE*>(RIManager.mShaders["skybox"]->psBytecode->GetBufferPointer()),
		RIManager.mShaders["skybox"]->psBytecode->GetBufferSize()
	};
	skySpherePsoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
	skySpherePsoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
	ThrowIfFailed(m_d3dDevice->CreateGraphicsPipelineState(&skySpherePsoDesc,
		IID_PPV_ARGS(&RIManager.mPSOs["Skybox"])));

	// 平面的PSO
	D3D12_GRAPHICS_PIPELINE_STATE_DESC texturePsoDesc = opaquePsoDesc;
	texturePsoDesc.VS =
	{
		reinterpret_cast<BYTE*>(RIManager.mShaders["texture"]->vsBytecode->GetBufferPointer()),
		RIManager.mShaders["texture"]->vsBytecode->GetBufferSize()
	};
	texturePsoDesc.PS =
	{
		reinterpret_cast<BYTE*>(RIManager.mShaders["texture"]->psBytecode->GetBufferPointer()),
		RIManager.mShaders["texture"]->psBytecode->GetBufferSize()
	};
	texturePsoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
	texturePsoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
	ThrowIfFailed(m_d3dDevice->CreateGraphicsPipelineState(&texturePsoDesc,
		IID_PPV_ARGS(&RIManager.mPSOs["Texture"])));

	//
	// PSO for shadow map pass.
	//
	D3D12_GRAPHICS_PIPELINE_STATE_DESC smapPsoDesc = opaquePsoDesc;
	smapPsoDesc.RasterizerState.DepthBias = 100000;
	smapPsoDesc.RasterizerState.DepthBiasClamp = 0.0f;
	smapPsoDesc.RasterizerState.SlopeScaledDepthBias = 1.0f;
	smapPsoDesc.pRootSignature = mRootSignature.Get();
	smapPsoDesc.VS =
	{
		reinterpret_cast<BYTE*>(RIManager.mShaders["ShadowOpaque"]->vsBytecode->GetBufferPointer()),
		RIManager.mShaders["ShadowOpaque"]->vsBytecode->GetBufferSize()
	};
	smapPsoDesc.PS =
	{
		reinterpret_cast<BYTE*>(RIManager.mShaders["ShadowOpaque"]->psBytecode->GetBufferPointer()),
		RIManager.mShaders["ShadowOpaque"]->psBytecode->GetBufferSize()
	};

	smapPsoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;//双面显示
	// Shadow map pass does not have a render target.
	smapPsoDesc.RTVFormats[0] = DXGI_FORMAT_UNKNOWN;
	smapPsoDesc.NumRenderTargets = 0;
	ThrowIfFailed(m_d3dDevice->CreateGraphicsPipelineState(&smapPsoDesc,
		IID_PPV_ARGS(&RIManager.mPSOs["shadow_opaque"])));
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
	XMStoreFloat4x4(&MainCamera.mProj, P);

	//resizeSwapChain(width(), height());
	ContinueFrames();
}

#pragma endregion

#pragma region HelperFuncs
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
	UpdateSubresources<1>(m_CommandList, defaultBuffer.Get(), uploadBuffer.Get(), 0, 0, 1, &subResourceData);

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

		ThrowIfFailed(m_CommandList->Close());
		ID3D12CommandList* cmdLists[] = { m_CommandList };
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

	// Init all modules using the device
	m_WorkSubmissionModule = std::make_unique<D3DModules::WorkSubmissionModule>(m_d3dDevice.Get());
	m_MemoryManagerModule = std::make_unique<D3DModules::MemoryManagerModule>(m_d3dDevice.Get());
	m_SynchronizationModule = std::make_unique<D3DModules::SynchronizationModule>(m_d3dDevice.Get());
	m_ResourceBindingModule = std::make_unique<D3DModules::ResourceBindingModule>(m_d3dDevice.Get());
	m_CudaManagerModule = std::make_unique<CudaManager>(m_dxgiFactory.Get(), m_d3dDevice.Get());
}
/// <summary>
/// Initialize:: 2 Create the Fance
/// </summary>
void QDirect3D12Widget::CreateFence()
{
	m_SynchronizationModule->CreateFence();
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
	// Create Command Queue
	m_CommandQueue = m_WorkSubmissionModule->CreateCommandQueue("direct", D3D12_COMMAND_LIST_TYPE_DIRECT);
	m_DirectCmdListAlloc = m_WorkSubmissionModule->CreateCommandListAllocator("main", D3D12_COMMAND_LIST_TYPE_DIRECT);
	m_CommandList = m_WorkSubmissionModule->CreateCommandList("main", D3D12_COMMAND_LIST_TYPE_DIRECT, "main");

	// Start off in a closed state.  This is because the first time we refer 
	// to the command list we will Reset it, and it needs to be closed before
	// calling Reset.
	m_CommandList->Close();	//重置命令列表前必须将其关闭
	m_CommandList->Reset(m_DirectCmdListAlloc, nullptr);
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
	ThrowIfFailed(m_dxgiFactory->CreateSwapChain(m_CommandQueue, &swapChainDesc, m_SwapChain.GetAddressOf()));
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

#pragma region Deprecated

#pragma endregion
