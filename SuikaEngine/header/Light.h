#pragma once

#include <Utility.h>
#include <Texture.h>

class FrameResource;
class RenderItemManager;
class PassConstants;

struct LightBasic
{
	DirectX::XMFLOAT3 Strength; // Light color
	float FalloffStart; // point/spot light only
	DirectX::XMFLOAT3 Direction;// directional/spot light only
	float FalloffEnd; // point/spot light only
	DirectX::XMFLOAT3 Position; // point/spot light only
	float SpotPower; // spot light only
};

struct Light {
	LightBasic basic;

	Light(bool isMain = false);

	void SetShadowMap(WritableTexture* texture)
	{
		mShadowMap = texture;
	}

	DirectX::XMFLOAT3 mLightPosW; // directional light only

	WritableTexture* mShadowMap = nullptr;

public:
	void StartDrawShadowMap(ID3D12GraphicsCommandList* mCommandList, FrameResource* mCurrFrameResource, RenderItemManager* RIManager);
	void EndDrawShadowMap(ID3D12GraphicsCommandList* mCommandList);
	void UpdateLightView();
	void UpdateShadowPassCB(FrameResource* mCurrFrameResource);

	XMFLOAT4X4 mShadowTransform = MathHelper::Identity4x4();

private:
	bool isMainLit = false;
	PassConstants* mShadowPassCB;

	float mLightNearZ = 0.0f;
	float mLightFarZ = 0.0f;

	XMFLOAT4X4 mLightView = MathHelper::Identity4x4();
	XMFLOAT4X4 mLightProj = MathHelper::Identity4x4();
};