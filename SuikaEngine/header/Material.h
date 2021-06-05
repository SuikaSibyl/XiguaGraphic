#pragma once

#include <string>
#include <MathHelper.h>
#include <PipelineSetting.h>

struct Material
{
	// Unique material name for lookup.
	std::string Name;

	// Index into constant buffer corresponding to this material.
	int MatCBIndex = -1;

	// Index into SRV heap for diffuse texture. Used in the texturing chapter.
	int DiffuseSrvHeapIndex = -1;

	// Dirty flag indicating the material has changed and we need to 
	// update the constant buffer. Because we have a material constant 
	// buffer for each FrameResource, we have to apply the update to each 
	// FrameResource. Thus, when we modify a material we should set 
	// NumFramesDirty = gNumFrameResources so that each frame resource 
	// gets the update.
	int NumFramesDirty = frameResourcesCount;

	// Material constant buffer data used for shading.
	DirectX::XMFLOAT4 DiffuseAlbedo = { 1.0f, 1.0f, 1.0f, 1.0f };
	DirectX::XMFLOAT3 FresnelR0 = { 0.01f, 0.01f, 0.01f };
	float Roughness = 0.25f;
	float Metalness = 0.25f;
	DirectX::XMFLOAT3 Emission = { 0, 0, 0 };
	DirectX::XMFLOAT4X4 MatTransform = MathHelper::Identity4x4();
};