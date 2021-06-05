#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <Debug.h>

#define checkCudaErrors(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      Debug::LogError(QString("ERROR: ")+QString(__FILE__)+QString(":")+QString::number(__LINE__));\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      fflush(stdout);\
      Debug::LogError(QString::fromStdString("code: " + std::to_string(error) + ", reason:" + cudaGetErrorString(error)));\
  }\
}\


#define checkCudaErrorsDev(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      fflush(stdout);\
  }\
}\
