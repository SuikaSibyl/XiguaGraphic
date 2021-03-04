
#pragma once
#include "Shader.h"
#include <Utility.h>

class Shader
{
public:
	Shader()
	{
		vsBytecode = CompileShader(L"shader\\color.hlsl", nullptr, "VS", "vs_5_0");
		psBytecode = CompileShader(L"shader\\color.hlsl", nullptr, "PS", "ps_5_0");
	}

	ComPtr<ID3DBlob> vsBytecode = nullptr;
	ComPtr<ID3DBlob> psBytecode = nullptr;

protected:
	ComPtr<ID3DBlob> CompileShader(
		const std::wstring& fileName,
		const D3D_SHADER_MACRO* defines,
		const std::string& enteryPoint,
		const std::string& target)
	{
		//�����ڵ���ģʽ����ʹ�õ��Ա�־
		UINT compileFlags = 0;
#if defined(DEBUG) || defined(_DEBUG)
		//�õ���ģʽ��������ɫ�� | ָʾ�����������Ż��׶�
		compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif // defined(DEBUG) || defined(_DEBUG)

		HRESULT hr = S_OK;

		ComPtr<ID3DBlob> byteCode = nullptr;
		ComPtr<ID3DBlob> errors;
		hr = D3DCompileFromFile(fileName.c_str(), //hlslԴ�ļ���
			defines,	//�߼�ѡ�ָ��Ϊ��ָ��
			D3D_COMPILE_STANDARD_FILE_INCLUDE,	//�߼�ѡ�����ָ��Ϊ��ָ��
			enteryPoint.c_str(),	//��ɫ������ڵ㺯����
			target.c_str(),		//ָ��������ɫ�����ͺͰ汾���ַ���
			compileFlags,	//ָʾ����ɫ���ϴ���Ӧ����α���ı�־
			0,	//�߼�ѡ��
			&byteCode,	//����õ��ֽ���
			&errors);	//������Ϣ

		if (errors != nullptr)
		{
			OutputDebugStringA((char*)errors->GetBufferPointer());
		}
		ThrowIfFailed(hr);

		return byteCode;
	}
};