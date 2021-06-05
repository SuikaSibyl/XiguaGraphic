#pragma once

#include <vector>
#include <DirectXMath.h>
#include <MeshGeometry.h>
#include <RenderItemManagment.h>

//执行波浪模拟的计算。在之后进行的模拟更新后，客户端必须将当前解决方案复制到顶点缓冲区中进行渲染。
class Waves
{
public:
    Waves(MeshGeometryHelper ihelper, int count) :helper(ihelper), mVertexCount(count) {};
    Waves(int m, int n, float dx, float dt, float speed, float damping);
    Waves(const Waves& rhs) = delete;
    Waves& operator=(const Waves& rhs) = delete;
    ~Waves() {};

    int RowCount()const;
    int ColumnCount()const;
    int VertexCount()const;
    int TriangleCount()const;
    float Width()const;
    float Depth()const;

    int mVertexCount = 0;
    MeshGeometryHelper helper;

    // 返回计算后的网格顶点坐标
    const DirectX::XMFLOAT3& Position(int i)const { return mCurrSolution[i]; }

    // 返回计算后的网格顶点法线
    const DirectX::XMFLOAT3& Normal(int i)const { return mNormals[i]; }

    // 返回计算后的网格顶点切线
    const DirectX::XMFLOAT3& TangentX(int i)const { return mTangentX[i]; }

    void Update(float dt);
    void Disturb(int i, int j, float magnitude);

    std::vector<Geometry::Vertex> vertex;

private:
    int mNumRows = 0;
    int mNumCols = 0;
    
    int mTriangleCount = 0;

    // Simulation constants we can precompute.
    float mK1 = 0.0f;
    float mK2 = 0.0f;
    float mK3 = 0.0f;

    float mTimeStep = 0.0f;
    float mSpatialStep = 0.0f;

    std::vector<DirectX::XMFLOAT3> mPrevSolution;
    std::vector<DirectX::XMFLOAT3> mCurrSolution;
    std::vector<DirectX::XMFLOAT3> mNormals;
    std::vector<DirectX::XMFLOAT3> mTangentX;
};