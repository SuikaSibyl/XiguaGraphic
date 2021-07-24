#pragma once
#include <Utility.h>
#include <Light.h>

static const int frameResourcesCount = 3;

struct ObjectConstants {
    DirectX::XMFLOAT4X4 world = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 texTransform = MathHelper::Identity4x4();

    //��ͬ����ȥ������Ӧ����
    UINT materialIndex = 0;//���ʵ�����
    UINT objPad0;	//ռλ
    UINT objPad1;	//ռλ
    UINT objPad2;	//ռλ
};

struct PassConstants
{
    // 0
    DirectX::XMFLOAT4X4 viewProj = MathHelper::Identity4x4();
    // 16
    XMFLOAT3 eyePos;
    float gTime = 0.0f;
    // 20
    XMFLOAT4 ambientLight;
    // 24
    DirectX::XMFLOAT4X4 gShadowTransform;
    // 40
    LightBasic light[16];
    // 56
    DirectX::XMFLOAT4X4 view = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 projection = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 viewInverse = MathHelper::Identity4x4();
};

struct MaterialData
{
    // 0
    XMFLOAT4 diffuseAlbedo = { 1.0f, 1.0f, 1.0f, 1.0f };//���ʷ�����
    // 4
    XMFLOAT3 fresnelR0 = { 0.01f, 0.01f, 0.01f };//RF(0)ֵ�������ʵķ�������
    float roughness = 0.25f;//���ʵĴֲڶ�
    // 8
    XMFLOAT4X4 matTransform = MathHelper::Identity4x4();//������λ�ƾ���
    // 24
    UINT diffuseMapIndex = 0;//������������
    UINT normalMapIndex;
    UINT extraMapIndex;
    UINT materialType;
    // 28
    //ռλ���������ݴ��ʱ����ռ��4λ
    float metalness = 0.25f;//���ʵĴֲڶ�
    XMFLOAT3 emission = { 0.0f, 0.1f, 0.0f };//RF(0)ֵ�������ʵķ�������
};