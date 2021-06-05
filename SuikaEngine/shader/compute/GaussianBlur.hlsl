cbuffer cbSettings : register(b0)
{   
	// We cannot have an array entry in a constant buffer that gets mapped onto
	// root constants, so list each element.  
    int gBlurRadius; //模糊半径

	// Support up to 11 blur weights.
    //11个权重（由于是根常量传过来的，所以不能用数组）
    float w0;
    float w1;
    float w2;
    float w3;
    float w4;
    float w5;
    float w6;
    float w7;
    float w8;
    float w9;
    float w10;    
};

static const int gMaxBlurRadius = 5;

Texture2D gInput : register(t0);    //输入的SRV纹理
RWTexture2D<float4> gOutput : register(u0);//输出的UAV纹理，注意模板中的数据类型

#define N 256  //一个线程组有256个线程
#define CacheSize (N + 2 * gMaxBlurRadius)     //共享内存大小

groupshared float4 gCache[CacheSize];   //声明共享内存

[numthreads(N, 1, 1)]//线程数定义（横向）
void HorzBlurCS(int3 groupThreadID : SV_GroupThreadID, //组内线程ID
                      int3 dispatchThreadID : SV_DispatchThreadID)//分派ID
{
    float weights[11] = { w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10 };//权重数组

    //对图像左侧的越界采样进行钳位操作
    if (groupThreadID.x < gBlurRadius)//如果是最左侧5个像素
    {
        //钳位左边的模糊半径个（5个）像素,x都是0
        int x = max(dispatchThreadID.x - gBlurRadius, 0); 
        //将input首位置的像素，钳位给左边的像素，并存入共享内存中
        gCache[groupThreadID.x] = gInput[int2(x, dispatchThreadID.y)];
    }
    
    //对图像右侧的越界采样进行钳位操作
    if (groupThreadID.x >= N - gBlurRadius)//如果是最右侧5个像素
    {
        //钳位右边的模糊半径个（5个）像素,x都是gInput.Length.x - 1,即最后一个像素索引
        int x = min(dispatchThreadID.x + gBlurRadius, gInput.Length.x - 1);
        //将input尾位置的像素，钳位给右边的像素，并存入共享内存中
        gCache[groupThreadID.x + 2 * gBlurRadius] = gInput[int2(x, dispatchThreadID.y)];
    }
    
    //中间的像素存入共享内存，注意，右侧的线程可能有剩余，所以要做钳位
    gCache[groupThreadID.x + gBlurRadius] = gInput[min(dispatchThreadID.xy, gInput.Length.xy - 1)];
    
    //等待所有线程完成任务
    GroupMemoryBarrierWithGroupSync();
    
    //
    //对每个像素进行模糊处理
    //
    float4 blurColor = float4(0, 0, 0, 0);//初始化颜色为黑色
	
    for (int i = -gBlurRadius; i <= gBlurRadius; ++i)//从-5到5循环遍历
    {
        int k = groupThreadID.x + gBlurRadius + i;//平移像素
        blurColor += weights[i + gBlurRadius] * gCache[k];//对“权重*纹素”求和（卷积）
    }
	
    gOutput[dispatchThreadID.xy] = blurColor;
    //gOutput[dispatchThreadID.xy] = gInput[dispatchThreadID.xy];

}

[numthreads(1, N, 1)] //线程数定义（纵向）
void VerticalBlurCS(int3 groupThreadID : SV_GroupThreadID, //组内线程ID
                      int3 dispatchThreadID : SV_DispatchThreadID)//分派ID
{
    float weights[11] = { w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10 }; //权重数组

    //对图像上侧的越界采样进行钳位操作
    if (groupThreadID.y < gBlurRadius)//如果是最上面5个像素
    {
        //钳位上面的模糊半径个（5个）像素,y都是0
        int y = max(dispatchThreadID.y - gBlurRadius, 0);
        //将input首位置的像素，钳位给上边的像素，并存入共享内存中
        gCache[groupThreadID.y] = gInput[int2(dispatchThreadID.x, y)];
    }
    
    //对图像下侧的越界采样进行钳位操作
    if (groupThreadID.y >= N - gBlurRadius)//如果是最下侧5个像素
    {
        //钳位下边的模糊半径个（5个）像素,y都是gInput.Length.y - 1,即最后一个像素索引
        int y = min(dispatchThreadID.y + gBlurRadius, gInput.Length.y - 1);
        //将input尾位置的像素，钳位给下边的像素，并存入共享内存中
        gCache[groupThreadID.y + 2 * gBlurRadius] = gInput[int2(dispatchThreadID.x, y)];
    }
    
    //中间的像素存入共享内存，注意，下侧的线程可能有剩余，所以要做钳位
    gCache[groupThreadID.y + gBlurRadius] = gInput[min(dispatchThreadID.xy, gInput.Length.xy - 1)];
    
    //等待所有线程完成任务
    GroupMemoryBarrierWithGroupSync();
    
    //
    //对每个像素进行模糊处理
    //
    float4 blurColor = float4(0, 0, 0, 0);
	
    for (int i = -gBlurRadius; i <= gBlurRadius; ++i)//从-5到5循环遍历
    {
        int k = groupThreadID.y + gBlurRadius + i; //平移像素
        blurColor += weights[i + gBlurRadius] * gCache[k]; //对“权重*纹素”求和（卷积）
    }
	
    gOutput[dispatchThreadID.xy] = blurColor;  
    //gOutput[dispatchThreadID.xy] = gInput[dispatchThreadID.xy];
}