#pragma once

#include <cuda_runtime.h>

__device__ const int kHardCodedOrderLimit = 4;

// Hardcoded spherical harmonic functions for low orders (l is first number
// and m is second number (sign encoded as preceeding 'p' or 'n')).
//
// As polynomials they are evaluated more efficiently in cartesian coordinates,
// assuming that @d is unit. This is not verified for efficiency.
__device__ float HardcodedSH00(const float3& d) {
    // 0.5 * sqrt(1/pi)
    return 0.282095;
}

__device__ float HardcodedSH1n1(const float3& d) {
    // -sqrt(3/(4pi)) * y
    return -0.488603 * d.y;
}

__device__ float HardcodedSH10(const float3& d) {
    // sqrt(3/(4pi)) * z
    return 0.488603 * d.z;
}

__device__ float HardcodedSH1p1(const float3& d) {
    // -sqrt(3/(4pi)) * x
    return -0.488603 * d.x;
}

__device__ float HardcodedSH2n2(const float3& d) {
    // 0.5 * sqrt(15/pi) * x * y
    return 1.092548 * d.x * d.y;
}

__device__ float HardcodedSH2n1(const float3& d) {
    // -0.5 * sqrt(15/pi) * y * z
    return -1.092548 * d.y * d.z;
}

__device__ float HardcodedSH20(const float3& d) {
    // 0.25 * sqrt(5/pi) * (-x^2-y^2+2z^2)
    return 0.315392 * (-d.x * d.x - d.y * d.y + 2.0 * d.z * d.z);
}

__device__ float HardcodedSH2p1(const float3& d) {
    // -0.5 * sqrt(15/pi) * x * z
    return -1.092548 * d.x * d.z;
}

__device__ float HardcodedSH2p2(const float3& d) {
    // 0.25 * sqrt(15/pi) * (x^2 - y^2)
    return 0.546274 * (d.x * d.x - d.y * d.y);
}

__device__ float HardcodedSH3n3(const float3& d) {
    // -0.25 * sqrt(35/(2pi)) * y * (3x^2 - y^2)
    return -0.590044 * d.y * (3.0 * d.x * d.x - d.y * d.y);
}

__device__ float HardcodedSH3n2(const float3& d) {
    // 0.5 * sqrt(105/pi) * x * y * z
    return 2.890611 * d.x * d.y * d.z;
}

__device__ float HardcodedSH3n1(const float3& d) {
    // -0.25 * sqrt(21/(2pi)) * y * (4z^2-x^2-y^2)
    return -0.457046 * d.y * (4.0 * d.z * d.z - d.x * d.x
        - d.y * d.y);
}

__device__ float HardcodedSH30(const float3& d) {
    // 0.25 * sqrt(7/pi) * z * (2z^2 - 3x^2 - 3y^2)
    return 0.373176 * d.z * (2.0 * d.z * d.z - 3.0 * d.x * d.x
        - 3.0 * d.y * d.y);
}

__device__ float HardcodedSH3p1(const float3& d) {
    // -0.25 * sqrt(21/(2pi)) * x * (4z^2-x^2-y^2)
    return -0.457046 * d.x * (4.0 * d.z * d.z - d.x * d.x
        - d.y * d.y);
}

__device__ float HardcodedSH3p2(const float3& d) {
    // 0.25 * sqrt(105/pi) * z * (x^2 - y^2)
    return 1.445306 * d.z * (d.x * d.x - d.y * d.y);
}

__device__ float HardcodedSH3p3(const float3& d) {
    // -0.25 * sqrt(35/(2pi)) * x * (x^2-3y^2)
    return -0.590044 * d.x * (d.x * d.x - 3.0 * d.y * d.y);
}

__device__ float HardcodedSH4n4(const float3& d) {
    // 0.75 * sqrt(35/pi) * x * y * (x^2-y^2)
    return 2.503343 * d.x * d.y * (d.x * d.x - d.y * d.y);
}

__device__ float HardcodedSH4n3(const float3& d) {
    // -0.75 * sqrt(35/(2pi)) * y * z * (3x^2-y^2)
    return -1.770131 * d.y * d.z * (3.0 * d.x * d.x - d.y * d.y);
}

__device__ float HardcodedSH4n2(const float3& d) {
    // 0.75 * sqrt(5/pi) * x * y * (7z^2-1)
    return 0.946175 * d.x * d.y * (7.0 * d.z * d.z - 1.0);
}

__device__ float HardcodedSH4n1(const float3& d) {
    // -0.75 * sqrt(5/(2pi)) * y * z * (7z^2-3)
    return -0.669047 * d.y * d.z * (7.0 * d.z * d.z - 3.0);
}

__device__ float HardcodedSH40(const float3& d) {
    // 3/16 * sqrt(1/pi) * (35z^4-30z^2+3)
    double z2 = d.z * d.z;
    return 0.105786 * (35.0 * z2 * z2 - 30.0 * z2 + 3.0);
}

__device__ float HardcodedSH4p1(const float3& d) {
    // -0.75 * sqrt(5/(2pi)) * x * z * (7z^2-3)
    return -0.669047 * d.x * d.z * (7.0 * d.z * d.z - 3.0);
}

__device__ float HardcodedSH4p2(const float3& d) {
    // 3/8 * sqrt(5/pi) * (x^2 - y^2) * (7z^2 - 1)
    return 0.473087 * (d.x * d.x - d.y * d.y)
        * (7.0 * d.z * d.z - 1.0);
}

__device__ float HardcodedSH4p3(const float3& d) {
    // -0.75 * sqrt(35/(2pi)) * x * z * (x^2 - 3y^2)
    return -1.770131 * d.x * d.z * (d.x * d.x - 3.0 * d.y * d.y);
}

__device__ float HardcodedSH4p4(const float3& d) {
    // 3/16*sqrt(35/pi) * (x^2 * (x^2 - 3y^2) - y^2 * (3x^2 - y^2))
    double x2 = d.x * d.x;
    double y2 = d.y * d.y;
    return 0.625836 * (x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2));
}

__device__ float EvalHardCodedSH(int l, int m, float3 dir)
{
    // Validate l and m here (don't do it generally since EvalSHSlow also
        // checks it if we delegate to that function).
    //CHECK(l >= 0, "l must be at least 0.");
    //CHECK(-l <= m && m <= l, "m must be between -l and l.");
    //CHECK(NearByMargin(dir.squaredNorm(), 1.0), "dir is not unit.");

    switch (l) {
    case 0:
        return HardcodedSH00(dir);
    case 1:
        switch (m) {
        case -1:
            return HardcodedSH1n1(dir);
        case 0:
            return HardcodedSH10(dir);
        case 1:
            return HardcodedSH1p1(dir);
        }
    case 2:
        switch (m) {
        case -2:
            return HardcodedSH2n2(dir);
        case -1:
            return HardcodedSH2n1(dir);
        case 0:
            return HardcodedSH20(dir);
        case 1:
            return HardcodedSH2p1(dir);
        case 2:
            return HardcodedSH2p2(dir);
        }
    case 3:
        switch (m) {
        case -3:
            return HardcodedSH3n3(dir);
        case -2:
            return HardcodedSH3n2(dir);
        case -1:
            return HardcodedSH3n1(dir);
        case 0:
            return HardcodedSH30(dir);
        case 1:
            return HardcodedSH3p1(dir);
        case 2:
            return HardcodedSH3p2(dir);
        case 3:
            return HardcodedSH3p3(dir);
        }
    case 4:
        switch (m) {
        case -4:
            return HardcodedSH4n4(dir);
        case -3:
            return HardcodedSH4n3(dir);
        case -2:
            return HardcodedSH4n2(dir);
        case -1:
            return HardcodedSH4n1(dir);
        case 0:
            return HardcodedSH40(dir);
        case 1:
            return HardcodedSH4p1(dir);
        case 2:
            return HardcodedSH4p2(dir);
        case 3:
            return HardcodedSH4p3(dir);
        case 4:
            return HardcodedSH4p4(dir);
        }
    }

    // This is unreachable given the CHECK's above but the compiler can't tell.
    return 0.0;
}

__device__ float3 ToVector(float phi, float theta) {
    double r = sinf(theta);
    return make_float3(r * cosf(phi), r * sinf(phi), cosf(theta));
}

__device__ float EvalSH(int l, int m, float phi, float theta) {
    // If using the hardcoded functions, switch to cartesian
    if (l <= kHardCodedOrderLimit) {
        return EvalHardCodedSH(l, m, ToVector(phi, theta));
    }
    else {
        // Stay in spherical coordinates since that's what the recurrence
        // version is implemented in
        //return EvalSHSlow(l, m, phi, theta);
    }
}

__device__ int GetIndex(int l, int m) {
    return l * (l + 1) + m;
}