#pragma once

#pragma comment(lib, "D3D12.lib")
#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")

#include <stdexcept>
#include <wrl.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <D3Dcompiler.h>

// ############################################################################
// ############################## Utils #######################################
// ############################################################################
#define ReleaseObject(object)                                                                 \
    if ((object) != Q_NULLPTR)                                                                \
    {                                                                                         \
        object->Release();                                                                    \
        object = Q_NULLPTR;                                                                   \
    }
#define ReleaseHandle(object)                                                                 \
    if ((object) != Q_NULLPTR)                                                                \
    {                                                                                         \
        CloseHandle(object);                                                                  \
        object = Q_NULLPTR;                                                                   \
    }

inline std::string HrToString(HRESULT hr)
{
    char s_str[64] = {};
    sprintf_s(s_str, "HRESULT of 0x%08X", static_cast<UINT>(hr));
    return std::string(s_str);
}

class HrException : public std::runtime_error
{
public:
    HrException(HRESULT hr)
        : std::runtime_error(HrToString(hr))
        , m_hr(hr)
    {
    }
    HRESULT Error() const { return m_hr; }

private:
    const HRESULT m_hr;
};

inline void ThrowIfFailed(HRESULT hr)
{
    if (FAILED(hr)) { throw HrException(hr); }
}

#define DXCall(func) ThrowIfFailed(func)
