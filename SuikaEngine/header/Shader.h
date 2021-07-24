#pragma once

#pragma comment(lib, "D3D12.lib")
#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")

#include <MathHelper.h>
#include <Utility.h>

class Shader
{
public:
	Shader(ID3D12Device* md3dDevice, std::wstring name, bool isComputeShader = false, std::string csfunc = "CS")
	{
		const D3D_SHADER_MACRO defines[] =
		{

			NULL, NULL
		};

		const D3D_SHADER_MACRO alphaTestDefines[] =
		{
			"ALPHA_TEST", "1",
			NULL, NULL
		};

		//2.输入布局描述和编译着色器字节码

		if (isComputeShader)
		{
			csBytecode = CompileShader(L"shader\\" + name + L".hlsl", nullptr, csfunc, "cs_5_1");
		}
		else
		{
			vsBytecode = CompileShader(L"shader\\" + name + L".hlsl", nullptr, "VS", "vs_5_1");
			psBytecode = CompileShader(L"shader\\" + name + L".hlsl", defines, "PS", "ps_5_1");
			psBytecodeAlphaTest = CompileShader(L"shader\\" + name + L".hlsl", alphaTestDefines, "PS", "ps_5_1");
		}

		// Create an array of D3D12_INPUT_ELEMENT_DESC
		//  + each element is a D3D12_INPUT_ELEMENT_DESC
		inputLayoutDesc =
		{
			  { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			  { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			  { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			  { "TRANSFER", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 32, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			  { "TRANSFER", 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 48, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			  { "TRANSFER", 2, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 64, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			  { "TRANSFER", 3, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 80, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			  { "THINESS", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 96, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			  { "THINESS", 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 112, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			  { "THINESS", 2, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 128, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			  { "THINESS", 3, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 144, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		};
	}

	// D3D12_INPUT_LAYOUT_DESC
	//  + describe the vertex data we use
	//	+ equals to array of D3D12_INPUT_ELEMENT_DESC
	//	+ used to fill out pso DESC -> InputLayout
	std::vector<D3D12_INPUT_ELEMENT_DESC> inputLayoutDesc;
	D3D12_INPUT_LAYOUT_DESC GetInputLayout()
	{
		return { this->inputLayoutDesc.data(), (UINT)this->inputLayoutDesc.size() };
	}

	ComPtr<ID3DBlob> vsBytecode = nullptr;
	ComPtr<ID3DBlob> psBytecode = nullptr;
	ComPtr<ID3DBlob> csBytecode = nullptr;
	ComPtr<ID3DBlob> psBytecodeAlphaTest = nullptr;

	unsigned long Release()
	{
		vsBytecode = nullptr;
		psBytecode = nullptr;
		csBytecode = nullptr;
		psBytecodeAlphaTest = nullptr;
		return 0;
	}

	~Shader()
	{
		Release();
	}
protected:
	ComPtr<ID3DBlob> CompileShader(
		const std::wstring& fileName,
		const D3D_SHADER_MACRO* defines,
		const std::string& enteryPoint,
		const std::string& target)
	{
		//若处于调试模式，则使用调试标志
		UINT compileFlags = 0;
#if defined(DEBUG) || defined(_DEBUG)
		//用调试模式来编译着色器 | 指示编译器跳过优化阶段
		compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif // defined(DEBUG) || defined(_DEBUG)

		HRESULT hr = S_OK;

		ComPtr<ID3DBlob> byteCode = nullptr;
		ComPtr<ID3DBlob> errors;
		hr = D3DCompileFromFile(fileName.c_str(), //hlsl源文件名
			defines,	//高级选项，指定为空指针
			D3D_COMPILE_STANDARD_FILE_INCLUDE,	//高级选项，可以指定为空指针
			enteryPoint.c_str(),	//着色器的入口点函数名
			target.c_str(),		//指定所用着色器类型和版本的字符串
			compileFlags,	//指示对着色器断代码应当如何编译的标志
			0,	//高级选项
			&byteCode,	//编译好的字节码
			&errors);	//错误信息

		if (errors != nullptr)
		{
			OutputDebugStringA((char*)errors->GetBufferPointer());
		}
		ThrowIfFailed(hr);

		return byteCode;
	}
};