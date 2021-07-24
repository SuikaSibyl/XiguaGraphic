#ifndef __PRT__
#define __PRT__

#include "Utility.hlsl"

#define EDGENUM_MAX 3
#define MAXORDER 3

struct float9
{
    float array[9];
};

//
//  |   verts:       projected polygon vertices |
//  |   numVerts:    NUM of polygon vertices    |
//
float SolidAngle_ProjectedPolygon(float3 verts[EDGENUM_MAX], int numVerts) {
    float sa = 0;
    float3 tmp1 = cross(verts[0], verts[numVerts-1]);
    float3 tmp2 = cross(verts[0], verts[1]);
    sa += acos(dot(tmp1, tmp2) / (length(tmp1) * length(tmp2)));
    tmp1 = cross(verts[1], verts[0]);
    tmp2 = cross(verts[1], verts[2]);
    sa += acos(dot(tmp1, tmp2) / (length(tmp1) * length(tmp2)));
    tmp1 = cross(verts[2], verts[1]);
    tmp2 = cross(verts[2], verts[0]);
    sa += acos(dot(tmp1, tmp2) / (length(tmp1) * length(tmp2)));
    sa -= (numVerts - 2) *PI;
    return sa;
}

// For a vertex e
// B update need to be executed by every vertex
float3 integralLocalBl(float a, float b, float gammaE)
{
    float acospbsin = a*cos(gammaE) + b*sin(gammaE);
    float asinmbcos = a*sin(gammaE) - b*cos(gammaE);

    float3 B_n;
    // B_0 = gamma_e
    B_n[0] = gammaE;
    // B_1 = a_e*sin(gamma_e) - b_e*cos(gamma_e) + b_e
    B_n[1] = asinmbcos + b;
    // Prepare for l = 2;
    // D_prev = D_{l−1}, D_1 = gamma_e
    float C_n = 0.5 * (asinmbcos * acospbsin + (a*a+b*b) * gammaE + a*b);
    B_n[2] = 1.5 * C_n - 0.5 * gammaE;

    return B_n;
}

// -------------------------------
// This function is computed for every Lobe Direction Shared
// ** Input:  omegaE, tangent, lambdaE **
// ** Output:  **
float3 calcLightZH(float3 omega, float3 omegaE[EDGENUM_MAX], float3 tangent[EDGENUM_MAX], float3 lambdaE[EDGENUM_MAX])
{
    float3 accumBl = float4(0,0,0,0);

    [unroll]
    for(int i = 0; i < 3; i++)
    {
        // ce = omega dot muE
        float cE = dot(omega, tangent[i]);
        // Accumulate the Bl on the i th edge
        float3 localBl = integralLocalBl(dot(omega, omegaE[i]), dot(omega, lambdaE[i]), acos(dot(omegaE[i], omegaE[(i+1)%3])));
        for (int n = 0; n < MAXORDER; n++) 
        {
            // Bl = Bl + cE * Ble
            accumBl[n] += localBl[n] * cE;
        }
    }

    // Return Sl
    float3 Sl;
    Sl[0] = 0.5 * accumBl[0];
    Sl[1] = 0.5 * accumBl[1];
    Sl[2] = dot(float2(0.416667, 0.166667), float2(accumBl[2], Sl[0]));
    return Sl;
}

//
//  |   x:   position of the vertex                 |
//  |   v:   the vertices of polygon light          |
//  |   M:   the number of edges of polygon light   |
//
float4x4 ComputeCoefficients(float3 x, float3 v[EDGENUM_MAX], int numVerts)
{
    float4x4 L_lm=0;

    // Stage1: Get the vertex projected to sphere
    // -----------------------------------
    float3 omegaE[EDGENUM_MAX];
    for(int i=0;i<EDGENUM_MAX;i++)
    {
        omegaE[i] = v[i]-x;
        if(omegaE[i].r==0 && omegaE[i].g==0 && omegaE[i].b==0)
            return L_lm;
        omegaE[i]=normalize(omegaE[i]);
    }

    // Test whether in the right direction
    float test = -omegaE[0].x*((omegaE[1].y-omegaE[0].y)*(omegaE[2].z-omegaE[0].z)-(omegaE[1].z-omegaE[0].z)*(omegaE[2].y-omegaE[0].y)) +
                + omegaE[0].y*((omegaE[1].x-omegaE[0].x)*(omegaE[2].z-omegaE[0].z)-(omegaE[1].z-omegaE[0].z)*(omegaE[2].x-omegaE[0].x)) +
                - omegaE[0].z*((omegaE[1].x-omegaE[0].x)*(omegaE[2].y-omegaE[0].y)-(omegaE[1].y-omegaE[0].y)*(omegaE[2].x-omegaE[0].x));
    
    if(test>-0.001)
    {
        return L_lm;
    }

    // Stage2: Precompute per edge
    // -----------------------------------
    float3 tangent[EDGENUM_MAX];
    tangent[0] = normalize(cross(omegaE[0], omegaE[1]));
    tangent[1] = normalize(cross(omegaE[1], omegaE[2]));
    tangent[2] = normalize(cross(omegaE[2], omegaE[0]));

    // lambda_e
    float3 lambdaE[EDGENUM_MAX];
    lambdaE[0] = cross(tangent[0], omegaE[0]);
    lambdaE[1] = cross(tangent[1], omegaE[1]);
    lambdaE[2] = cross(tangent[2], omegaE[2]);

    // Stage4: For each rotated ZH lobe, calc the ZH coefficient 
    // -----------------------------------

    // In total 17 = 2N + 1 (N = 8) lobe directions
    // Each direction is shared by all bands
    // wxy is actually all the ZHs directed to a single lobe direction

    float3 SlDir0 = calcLightZH((float3(0.866025, -0.500001, -0.000004)),  omegaE, tangent, lambdaE);
    float3 SlDir1 = calcLightZH((float3(-0.759553, 0.438522, -0.480394)),  omegaE, tangent, lambdaE);
    float3 SlDir2 = calcLightZH((float3(-0.000002, 0.638694, 0.769461)),   omegaE, tangent, lambdaE);
    float3 SlDir3 = calcLightZH((float3(-0.000004, -1.000000, -0.000004)), omegaE, tangent, lambdaE);
    float3 SlDir4 = calcLightZH((float3(-0.000007, 0.000003, -1.000000)),  omegaE, tangent, lambdaE);
    float3 SlDir5 = calcLightZH((float3(-0.000002, -0.638694, 0.769461)),  omegaE, tangent, lambdaE);
    float3 SlDir6 = calcLightZH((float3(-0.974097, 0.000007, -0.226131)),  omegaE, tangent, lambdaE);

    // Stage5: ZH Factorization: Get Light SH Coeff from ZH Coeff
    // -----------------------------------
    // Light coeffs are linear combination of Directed ZH

    // band-0
    L_lm[0][0] = 0.282095 * SolidAngle_ProjectedPolygon(omegaE, EDGENUM_MAX);

    // band-1
    L_lm[0][1] = dot(float3(1.18466498143, 0.7074501249, 0.441684699), float3(SlDir0[0], SlDir1[0], SlDir2[0]));
    L_lm[0][2] = dot(float2(-0.8920539, -1.0170995749), float2(SlDir0[0], SlDir1[0]));
    L_lm[0][3] = dot(float3(1.07469776, 1.22534357, 0.76501818), float3(SlDir0[0], SlDir1[0], SlDir2[0]));

    // band-2 / 5-SH
    L_lm[1][0] = dot(float2(-0.728370563, -0.3641886), float2(SlDir3[1], SlDir4[1]));
    L_lm[1][1] = dot(float3(-0.7677286, 0.99806346, 0.427833122), float3(SlDir0[1], SlDir1[1], SlDir2[1])) + 
                 dot(float2(-0.1745145, -0.4836308), float2(SlDir3[1], SlDir4[1]));
    L_lm[1][2] = 0.6307831 * SlDir4[1];
    L_lm[1][3] = dot(float3(0.741040075, -0.302295225, -0.438750224), float3(SlDir2[1], SlDir3[1], SlDir4[1]));
    L_lm[2][0] = dot(float3(-0.84103184, -0.420522176, -0.6307806), float3(SlDir0[1], SlDir3[1], SlDir4[1]));

    // band-3 / 7-SH
    L_lm[2][1] = dot(float3(-0.85515113, 0.7731073459, -0.6342028638), float3(SlDir0[2], SlDir2[2], SlDir3[2]))
               + dot(float3(-0.3275784995, -0.0749077, -1.0214063487), float3(SlDir4[2], SlDir5[2], SlDir6[2]));
    L_lm[2][2] = dot(float3(-0.613941102, 0.01873014256, -0.613937359), float3(SlDir2[2], SlDir4[2], SlDir5[2]));
    L_lm[2][3] = dot(float4(-1.407351116699, 0.697809744499, -1.0437454, -0.6978231953), 
                     float4(SlDir0[2], SlDir2[2], SlDir3[2], SlDir5[2]));
    L_lm[3][0] = -0.74635300019405178 * SlDir4[2];
    L_lm[3][1] = dot(float3(0.4028760698, 0.281276817357, 0.209931796464849), float3(SlDir2[2], SlDir3[2], SlDir5[2]));
    L_lm[3][2] = dot(float3(-0.16255861, 1.20435457, -0.0321461186), float3(SlDir0[2], SlDir1[2], SlDir2[2]))
               + dot(float3(-0.8071183418, 0.544847474, -0.67678489), float3(SlDir3[2], SlDir4[2], SlDir5[2]));
    L_lm[3][3] = dot(float3(-0.31207144274, 0.7814676556, 0.312076866), float3(SlDir2[2], SlDir3[2], SlDir5[2]));

    return L_lm;
}

#endif