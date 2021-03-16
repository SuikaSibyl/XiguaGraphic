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
	Shader(ComPtr<ID3D12Device> md3dDevice)
	{
		//2.输入布局描述和编译着色器字节码

		vsBytecode = CompileShader(L"shader\\Color.hlsl", nullptr, "VS", "vs_5_0");
		psBytecode = CompileShader(L"shader\\Color.hlsl", nullptr, "PS", "ps_5_0");

		// Create an array of D3D12_INPUT_ELEMENT_DESC
		//  + each element is a D3D12_INPUT_ELEMENT_DESC
		inputLayoutDesc =
		{
			  { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			  { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			  { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
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

	ComPtr<ID3D12RootSignature> rootSignature;

	ComPtr<ID3DBlob> vsBytecode = nullptr;
	ComPtr<ID3DBlob> psBytecode = nullptr;
	ComPtr<ID3DBlob> serializedRootSig = nullptr;
	ComPtr<ID3DBlob> errorBlob = nullptr;
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

	void BuildRootSignature(ComPtr<ID3D12Device> md3dDevice)
	{
		//根参数可以是描述符表、根描述符、根常量
		CD3DX12_ROOT_PARAMETER slotRootParameter[1];
		//创建由单个CBV所组成的描述符表
		CD3DX12_DESCRIPTOR_RANGE cbvTable;
		cbvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, //描述符类型
			1, //描述符数量
			0);//描述符所绑定的寄存器槽号
		slotRootParameter[0].InitAsDescriptorTable(1, &cbvTable);
		//根签名由一组根参数构成
		CD3DX12_ROOT_SIGNATURE_DESC rootSig(1, //根参数的数量
			slotRootParameter, //根参数指针
			0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
		//用单个寄存器槽来创建一个根签名，该槽位指向一个仅含有单个常量缓冲区的描述符区域
		HRESULT hr = D3D12SerializeRootSignature(&rootSig, D3D_ROOT_SIGNATURE_VERSION_1, &serializedRootSig, &errorBlob);

		if (errorBlob != nullptr)
		{
			OutputDebugStringA((char*)errorBlob->GetBufferPointer());
		}
		ThrowIfFailed(hr);

		ThrowIfFailed(md3dDevice->CreateRootSignature(0,
			serializedRootSig->GetBufferPointer(),
			serializedRootSig->GetBufferSize(),
			IID_PPV_ARGS(&rootSignature)));
	}
};