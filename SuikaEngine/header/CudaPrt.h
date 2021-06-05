#pragma once

#ifndef __CUDA_PRT_H_
#define __CUDA_PRT_H_

#include <iostream>
#include "linear_algebra.h"
#include "geometry.h"

#include <fstream>
#include <iostream>
#include <string>

class PRTransfer
{
public:
	CuVertex* pPrtVertices;
	float* pTransferData;
	float* pVertexNormal;

	PRTransfer(CuVertex* vertices, int vnum, int order = 3)
	{
		pPrtVertices = vertices;
		mVerticesNum = vnum;
		mSHOrder = order;
		pTransferData = new float[TransferNum()];
		pVertexNormal = new float[mVerticesNum * 6];
		for (int i = 0; i < vnum; i++)
		{
			pVertexNormal[i * 6 + 0] = vertices[i].x;
			pVertexNormal[i * 6 + 1] = vertices[i].y;
			pVertexNormal[i * 6 + 2] = vertices[i].z;
			pVertexNormal[i * 6 + 3] = vertices[i]._normal.x;
			pVertexNormal[i * 6 + 4] = vertices[i]._normal.y;
			pVertexNormal[i * 6 + 5] = vertices[i]._normal.z;
		}
	}

	PRTransfer(int vnum, int order = 3)
	{
		mVerticesNum = vnum;
		mSHOrder = order;
		pTransferData = new float[TransferNum()];
	}

	int VertexSize()
	{
		return mVerticesNum * 8 * sizeof(float);
	}

	int VertexNum()
	{
		return mVerticesNum;
	}

	int TransferNum()
	{
		return (1 + mSHOrder) * (1 + mSHOrder) * mVerticesNum;
	}

	int TransferSize()
	{
		return (1 + mSHOrder) * (1 + mSHOrder) * mVerticesNum * sizeof(float);
	}

	bool LoadFromFile(std::string name)
	{
		std::ifstream ifs(name, std::ios::binary | std::ios::in);
		if (!ifs)
		{
			return false;
		}
		ifs.read((char*)pTransferData, TransferSize());
		ifs.close();
		return true;
	}

	void WriteToFile(std::string name)
	{
		std::ofstream  ofs(name, std::ios::binary | std::ios::out);
		ofs.write((const char*)pTransferData, TransferSize());
		ofs.close();
	}

	unsigned int mVerticesNum;
	unsigned int mSHOrder;
};

void RunPrecomputeTransfer(PRTransfer* prt);

#endif