#pragma once
#include <PipelineSetting.h>
#include <Platform/DirectX12/FrameResources.h>
#include <Texture.h>
#include <Shader.h>
#include <memory>
#include <Platform/DirectX12/StructuredBuffer.h>

namespace D3DModules
{
	class ComputeInstance
	{
	public:
		ComputeInstance(ID3D12Device* d3dDevice, std::wstring shadername, ID3D12RootSignature* signature, std::string csfunc = "CS") :
			m_Rootsignature(signature)
		{
			m_ComputeShader = new Shader(d3dDevice, shadername, true, csfunc);
			CreateComputePSO(d3dDevice);
		}

		~ComputeInstance()
		{
			delete m_ComputeShader;
			m_Rootsignature = nullptr;
			m_PSO = nullptr;
			return;
		}

		void Execute(ID3D12GraphicsCommandList* cmdList, StructuredBuffer* buffer, int lightNum)
		{
			cmdList->SetPipelineState(m_PSO.Get());//ִ�к��������ɫ��

			std::vector<float> weights = { .1,.2,.4,.2,.1 };
			cmdList->SetComputeRootSignature(m_Rootsignature);//�󶨴���������ɫ���ĸ�ǩ��
			cmdList->SetComputeRoot32BitConstants(0, (UINT)weights.size(), weights.data(), 1);//���ø�����
			cmdList->SetComputeRoot32BitConstants(0, 1, &lightNum, 0);//���ø�����
			cmdList->SetComputeRootShaderResourceView(1,//����������
				light_addr);//����Դ��ַ

			cmdList->SetComputeRootDescriptorTable(2, buffer->gpuUav());//UAV�󶨸���������
			cmdList->Dispatch(4, 4, 4);//�����߳���
		}

		void Execute(ID3D12GraphicsCommandList* cmdList, WritableTexture* input)
		{
			input->ChangeResourceState(
				cmdList,
				D3D12_RESOURCE_STATE_GENERIC_READ);

			ua1->ChangeResourceState(
				cmdList,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

			cmdList->SetPipelineState(m_PSO.Get());//ִ�к��������ɫ��

			std::vector<float> weights = { .1,.2,.4,.2,.1 };
			int blurRadius = 2;
			cmdList->SetComputeRootSignature(m_Rootsignature);//�󶨴���������ɫ���ĸ�ǩ��
			cmdList->SetComputeRoot32BitConstants(0, (UINT)weights.size(), weights.data(), 1);//���ø�����
			cmdList->SetComputeRoot32BitConstants(0, 1, &blurRadius, 0);//���ø�����
			cmdList->SetComputeRootDescriptorTable(1, input->gpuSrv());//SRV�󶨸���������
			cmdList->SetComputeRootDescriptorTable(2, ua1->gpuUav());//UAV�󶨸���������
			UINT numGroupsX = (UINT)ceilf(ua1->Width() / 256.0f);//X������߳�������
			UINT numGroupsY = (UINT)ceilf(ua1->Height() / 256.0f);//Y������߳�������
			cmdList->Dispatch(numGroupsX, ua1->Height(), 1);//�����߳���
		}

		WritableTexture* ua1;
		WritableTexture* ua2;
		D3D12_GPU_VIRTUAL_ADDRESS light_addr;

		CD3DX12_GPU_DESCRIPTOR_HANDLE srv;
		CD3DX12_GPU_DESCRIPTOR_HANDLE uav;

	private:
		void CreateComputePSO(ID3D12Device* d3dDevice);

		Shader* m_ComputeShader = nullptr;
		ID3D12RootSignature* m_Rootsignature = nullptr;
		ComPtr<ID3D12PipelineState> m_PSO = nullptr;
	};
	class ResourceBindingModule;

	class ComputeSubmodule
	{
	public:
		ComputeSubmodule(ResourceBindingModule* RBModule) :main(RBModule)
		{

		}

		ComputeInstance* ComputeIns(std::string name)
		{
			return m_ComputeInstances[name].get();
		}

		void BuildPostProcessRootSignature();
		void BuildPolygonSHRootSignature();

		void CreateComputeInstance(std::string name, std::string signature, std::wstring shadername, std::string csfunc = "CS");

	private:
		ResourceBindingModule* main;
		std::unordered_map<std::string, std::unique_ptr<ComputeInstance>> m_ComputeInstances;
		std::unordered_map<std::string, ComPtr<ID3D12RootSignature>> m_CSRootSignatures;
	};

	class ResourceBindingModule
	{
		friend class ComputeSubmodule;
		// ---------------------------------------
		// Binding Resources
		// ---------------------------------------
	public:
		ResourceBindingModule(ID3D12Device* d3dDevice);
		ComputeSubmodule ComputeSub;

		void CreateBuffer(uint sizeInBytes);
	private:
		ID3D12Device* m_d3dDevice;

	public:
		PassConstants& GetPassConstants();
		ObjectConstants& GetObjectConstants();

		void UpdatePassConstants(FrameResource* currFrameResource, int i = 0);
		void BindPassConstants(ID3D12GraphicsCommandList* commandList, FrameResource* currFrameResource, int i = 0);
		void UpdateObjectConstants(FrameResource* currFrameResource);

	private:
		PassConstants passConstants;
		ObjectConstants objConstants;

		UINT passCBByteSize = Utils::CalcConstantBufferByteSize(sizeof(PassConstants));

		// ---------------------------------------
		// Root Signatures
		// ---------------------------------------
	public:
		std::unordered_map<std::string, ComPtr<ID3D12RootSignature>> m_RootSignatures;
	private:

		// ---------------------------------------
		// PSO
		// ---------------------------------------
	public:
	private:
		std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> m_PSOs;
	};
}