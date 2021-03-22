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
    DirectX::XMFLOAT4X4 viewProj = MathHelper::Identity4x4();
    XMFLOAT3 eyePos;
    float gTime = 0.0f;
    XMFLOAT4 ambientLight;
    Light light[16];
};

struct MaterialData
{
    XMFLOAT4 diffuseAlbedo = { 1.0f, 1.0f, 1.0f, 1.0f };//���ʷ�����
    XMFLOAT3 fresnelR0 = { 0.01f, 0.01f, 0.01f };//RF(0)ֵ�������ʵķ�������
    float roughness = 0.25f;//���ʵĴֲڶ�
    XMFLOAT4X4 matTransform = MathHelper::Identity4x4();//������λ�ƾ���

    UINT diffuseMapIndex = 0;//������������
    //ռλ���������ݴ��ʱ����ռ��4λ
    UINT matPad0;
    UINT matPad1;
    UINT matPad2;
};