#include <Precompiled.h>
#include <Light.h>
#include <Platform/DirectX12/FrameResources.h>
#include <RenderItemManagment.h>
#include <PipelineSetting.h>

Light::Light(bool isMain)
{
	isMainLit = isMain;
	mShadowPassCB = new PassConstants();
}

void Light::StartDrawShadowMap(ID3D12GraphicsCommandList* mCommandList, FrameResource* mCurrFrameResource, RenderItemManager* RIManager)
{
	if (mShadowMap == nullptr)
	{
		Debug::LogError("This Light do not have ShadowMap");
		return;
	}

	mCommandList->RSSetViewports(1, &mShadowMap->Viewport());
	mCommandList->RSSetScissorRects(1, &mShadowMap->ScissorRect());

	// Change to DEPTH_WRITE.
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
		mShadowMap->Resource(),
		D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_DEPTH_WRITE));

	// Clear the back buffer and depth buffer.
	mCommandList->ClearDepthStencilView(mShadowMap->Dsv(),
		D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);

	// Set null render target because we are only going to draw to
	// depth buffer.  Setting a null render target will disable color writes.
	// Note the active PSO also must specify a render target count of 0.
	mCommandList->OMSetRenderTargets(0, nullptr, false, &mShadowMap->Dsv());

	//分别设置PSO并绘制对应渲染项
	mCommandList->SetPipelineState(RIManager->mPSOs["shadow_opaque"].Get());
}

void Light::EndDrawShadowMap(ID3D12GraphicsCommandList* mCommandList)
{
	if (mShadowMap == nullptr)
	{
		Debug::LogError("This Light do not have ShadowMap");
		return;
	}

	// Change back to GENERIC_READ so we can read the texture in a shader.
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(mShadowMap->Resource(),
		D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_GENERIC_READ));
}

void Light::UpdateLightView()
{
	DirectX::BoundingSphere mSceneBounds;
	mSceneBounds.Center = XMFLOAT3(0.0f, 0.0f, 0.0f);
	mSceneBounds.Radius = 80;

	// Only the first "main" light casts a shadow.
	XMVECTOR lightDir = XMLoadFloat3(&basic.Direction);
	XMVECTOR lightPos = -2.0f * mSceneBounds.Radius * lightDir;
	XMVECTOR targetPos = XMLoadFloat3(&mSceneBounds.Center);
	XMVECTOR lightUp = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
	XMMATRIX lightView = XMMatrixLookAtLH(lightPos, targetPos, lightUp);

	XMStoreFloat3(&mLightPosW, lightPos);

	// Transform bounding sphere to light space.
	XMFLOAT3 sphereCenterLS;
	XMStoreFloat3(&sphereCenterLS, XMVector3TransformCoord(targetPos, lightView));

	// Ortho frustum in light space encloses scene.
	float l = sphereCenterLS.x - mSceneBounds.Radius;
	float b = sphereCenterLS.y - mSceneBounds.Radius;
	float n = sphereCenterLS.z - mSceneBounds.Radius;
	float r = sphereCenterLS.x + mSceneBounds.Radius;
	float t = sphereCenterLS.y + mSceneBounds.Radius;
	float f = sphereCenterLS.z + mSceneBounds.Radius;

	mLightNearZ = n;
	mLightFarZ = f;

	XMMATRIX lightProj = XMMatrixOrthographicOffCenterLH(l, r, b, t, n, f);

	// Transform NDC space [-1,+1]^2 to texture space [0,1]^2
	XMMATRIX T(
		0.5f, 0.0f, 0.0f, 0.0f,
		0.0f, -0.5f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.5f, 0.5f, 0.0f, 1.0f);

	XMMATRIX S = lightView * lightProj * T;
	XMStoreFloat4x4(&mLightView, lightView);
	XMStoreFloat4x4(&mLightProj, lightProj);
	XMStoreFloat4x4(&mShadowTransform, S);
}

void Light::UpdateShadowPassCB(FrameResource* mCurrFrameResource)
{
	XMMATRIX view = XMLoadFloat4x4(&mLightView);
	XMMATRIX proj = XMLoadFloat4x4(&mLightProj);

	XMMATRIX viewProj = view * proj;
	// Update the constant buffer with the latest worldViewProj matrix.
	XMStoreFloat4x4(&mShadowPassCB->viewProj, XMMatrixTranspose(viewProj));

	auto currPassCB = mCurrFrameResource->passCB.get();
	currPassCB->CopyData(1, *mShadowPassCB);
}