#include "Utility.hlsl"


#define EDGENUM_MAX 3

struct float9
{
    float array[9];
};

float9 Legendre(float x)
{
	float9 P;

    P.array[0] = 1;
	P.array[1] = x;
    P.array[2] = (3*x*P.array[1] - 1*P.array[0])/2;
    P.array[3] = (5*x*P.array[2] - 2*P.array[1])/3;
    P.array[4] = (7*x*P.array[3] - 3*P.array[2])/4;
    P.array[5] = (9*x*P.array[4] - 4*P.array[3])/5;
    P.array[6] = (11*x*P.array[5] - 5*P.array[4])/6;
    // P[7] = (13*x*P[6] - 6*P[5])/7;

    // vec4 texFetch = texture(LUT_L2345, x);

    // P[2] = texFetch.x;
    // P[3] = texFetch.y;
    // P[4] = texFetch.z;
    // P[5] = texFetch.w;

    // vec4 texFetch2 = texture(LUT_L6789, x);

    // P[6] = texFetch2.x;

    // if (x < 0) {
    //     P[3] = -P[3];
    //     P[5] = -P[5];
    // }

    return P;
}

float9 boundary(float a, float b, float x, int maxN) {

    float9 B_n;

    float z = a*cos(x) + b*sin(x);
    float tmp1 = a*sin(x) - b*cos(x);
    float tmp2 = a*a+b*b-1;

    float9 P = Legendre(z);
    float9 Pa = Legendre(a);

    B_n.array[0] = x;
    B_n.array[1] = tmp1 + b;

	float D_next = 3 * B_n.array[1];
	float D_prev = x;

    for (int i = 2; i < maxN; i++) {
        float sf = 1.0/i;

        float C_n = (tmp1 * P.array[i-1]) + (tmp2 * D_prev) + ((i-1) * B_n.array[i-2]) + (b * Pa.array[i-1]);
        C_n *= sf;

		B_n.array[i] = (2*i-1) * C_n - (i - 1) * B_n.array[i - 2];
		B_n.array[i] *= sf;

		float temp = D_next;
		D_next = (2 * i + 1) * B_n.array[i] + D_prev;
		D_prev = temp;
    }

    return B_n.array;
}

float9 evalLight(float3 dir, float3 verts[EDGENUM_MAX], float3 gam[EDGENUM_MAX], float3 gamP[EDGENUM_MAX],
                    int maxN, int numVerts) {

    float9 total;

	float9 bound = boundary(dot(dir, verts[0]), dot(dir, gamP[0]),
		acos(dot(verts[0], verts[1])), maxN);
	for (int n = 0; n < maxN; n++) {
		total.array[n] = bound.array[n] * dot(dir, gam[0]);
	}

    // i = 1
    bound = boundary(dot(dir, verts[1]), dot(dir, gamP[1]),
                                   acos(dot(verts[1], verts[2])), maxN);
    for (int n = 0; n < maxN; n++) {
        total.array[n] += bound.array[n] * dot(dir, gam[1]);
    }

    // i = 2
    bound = boundary(dot(dir, verts[2]), dot(dir, gamP[2]),
                                   acos(dot(verts[2], verts[3 % numVerts])), maxN);
    for (int n = 0; n < maxN; n++) {
        total.array[n] += bound.array[n] * dot(dir, gam[2]);
    }

    // i = 3
    if (numVerts >= 4) {
        bound = boundary(dot(dir, verts[3]), dot(dir, gamP[3]),
                                   acos(dot(verts[3], verts[4 % numVerts])), maxN);
        for (int n = 0; n < maxN; n++) {
            total.array[n] += bound.array[n] * dot(dir, gam[3]);
        }
    }

    if (numVerts >= 5) {
        bound = boundary(dot(dir, verts[4]), dot(dir, gamP[4]),
                                   acos(dot(verts[4], verts[5 % numVerts])), maxN);
        for (int n = 0; n < maxN; n++) {
            total.array[n] += bound.array[n] * dot(dir, gam[4]);
        }
    }

    float9 surf;
    surf.array[1] = 0.5 * total.array[0];
    surf.array[2] = 0.5 * total.array[1];
    surf.array[3] = dot(float2(0.416667, 0.166667), float2(total.array[2], surf.array[1]));
    surf.array[4] = dot(float2(0.35, 0.3), float2(total.array[3], surf.array[2]));
    surf.array[5] = dot(float2(0.3, 0.4), float2(total.array[4], surf.array[3]));
    surf.array[6] = dot(float2(0.261905, 0.476190), float2(total.array[5], surf.array[4]));
    surf.array[7] = dot(float2(0.232143, 0.535714), float2(total.array[6], surf.array[5]));
    surf.array[8] = dot(float2(0.208333, 0.583333), float2(total.array[7], surf.array[6]));

    
    return surf;
}
//
//  |   verts:       projected polygon vertices |
//  |   numVerts:    NUM of polygon vertices    |
//
float SolidAngle_ProjectedPolygon(float3 verts[EDGENUM_MAX], int numVerts) {
    float sa = 0;
    float3 tmp1 = cross(verts[0], verts[numVerts-1]);
    float3 tmp2 = cross(verts[0], verts[1]);
    sa += acos(dot(tmp1, tmp2) / (length(tmp1) * length(tmp2)));

    // Polygon will be at least a triangle
    // i = 1
    tmp1 = cross(verts[1], verts[0]);
    tmp2 = cross(verts[1], verts[2]);
    sa += acos(dot(tmp1, tmp2) / (length(tmp1) * length(tmp2)));

    // i = 2
    tmp1 = cross(verts[2], verts[1]);
    tmp2 = cross(verts[2], verts[3 % numVerts]);
    sa += acos(dot(tmp1, tmp2) / (length(tmp1) * length(tmp2)));

    if (numVerts >= 4) {
        tmp1 = cross(verts[3], verts[2]);
        tmp2 = cross(verts[3], verts[4 % numVerts]);
        sa += acos(dot(tmp1, tmp2) / (length(tmp1) * length(tmp2)));
    }
    if (numVerts >= 5) {
        tmp1 = cross(verts[4], verts[3]);
        tmp2 = cross(verts[4], verts[0]);   // for now let max vertices be 5
        sa += acos(dot(tmp1, tmp2) / (length(tmp1) * length(tmp2)));
    }

    sa -= (numVerts - 2) *PI;
    return sa;
}

//
//  |   x:   position of the vertex                 |
//  |   v:   the vertices of polygon light          |
//  |   M:   the number of edges of polygon light   |
//
float4x4 ComputeCoefficients(float3 x, float3 v[EDGENUM_MAX], int numVerts)
{
    float4x4 L_lm;

    // Stage1: Get the vertex projected to sphere
    // -----------------------------------
    float3 L[EDGENUM_MAX];
    for(int i=0;i<EDGENUM_MAX;i++)
    {
        L[i]=normalize(v[i]-x);
    }

    // Stage2: Precompute per edge
    // -----------------------------------
    // float3 lambda[EDGENUM_MAX];
    // float3 miu[EDGENUM_MAX];
    // float gamma[EDGENUM_MAX];
    // for(int i=0;i<EDGENUM_MAX;i++)
    // {
    //     int next = (i==EDGENUM_MAX-1)?(i+1):EDGENUM_MAX;
    //     // Get the Frame
    //     lambda[i]= cross(normalize(cross(L[i],L[next])), L[i]);
    //     miu[i]=cross(L[i],L[next]);
    //     // Angle of edge e
    //     gamma[i]=acos(dot(L[i],L[next]));
    // }
    float3 G[EDGENUM_MAX];
    G[0] = normalize(cross(L[0], L[1]));
    G[1] = normalize(cross(L[1], L[2]));
    G[2] = normalize(cross(L[2], L[3 % numVerts]));

    float3 Gp[EDGENUM_MAX];
    Gp[0] = cross(G[0], L[0]);
    Gp[1] = cross(G[1], L[1]);
    Gp[2] = cross(G[2], L[2]);

    if (numVerts >= 4) {
        G[3] = normalize(cross(L[3], L[4 % numVerts]));
        Gp[3] = cross(G[3], L[3]);
    }
    if (numVerts >= 5) {
        G[4] = normalize(cross(L[4], L[5 % numVerts]));
        Gp[4] = cross(G[4], L[4]);
    }

    // Stage3: Init S0 to Solid Angle
    // -----------------------------------
    float SA = SolidAngle_ProjectedPolygon(L, EDGENUM_MAX);
    int max_order = 3;

    // Stage4: For each lobe direction L, iteration
    // -----------------------------------

    // Order-8, get 16 direction
    // For each direction, direct all the 9 ZHs
    // wxy is actually all the ZHs directed to a single lobe direction

    float9 w20 = evalLight((float3(0.866025, -0.500001, -0.000004)), L, G, Gp, max_order, numVerts);
    float9 w21 = evalLight((float3(-0.759553, 0.438522, -0.480394)), L, G, Gp, max_order, numVerts);
    float9 w22 = evalLight((float3(-0.000002, 0.638694, 0.769461)), L, G, Gp, max_order, numVerts);
    float9 w23 = evalLight((float3(-0.000004, -1.000000, -0.000004)), L, G, Gp, max_order, numVerts);
    float9 w24 = evalLight((float3(-0.000007, 0.000003, -1.000000)), L, G, Gp, max_order, numVerts);
    float9 w25 = evalLight((float3(-0.000002, -0.638694, 0.769461)), L, G, Gp, max_order, numVerts);
    float9 w26 = evalLight((float3(-0.974097, 0.000007, -0.226131)), L, G, Gp, max_order, numVerts);
    float9 w27 = evalLight((float3(-0.000003, 0.907079, -0.420960)), L, G, Gp, max_order, numVerts);
    float9 w28 = evalLight((float3(-0.960778, 0.000007, -0.277320)), L, G, Gp, max_order, numVerts);
    float9 w29 = evalLight((float3(-0.000003, -0.907079, -0.420960)), L, G, Gp, max_order, numVerts);
    float9 w30 = evalLight((float3(-0.451627, -0.451622, 0.769461)), L, G, Gp, max_order, numVerts);
    float9 w31 = evalLight((float3(-0.000003, -0.806136, -0.591730)), L, G, Gp, max_order, numVerts);
    float9 w32 = evalLight((float3(0.767864, 0.000000, 0.640613)), L, G, Gp, max_order, numVerts);
    float9 w33 = evalLight((float3(-0.000003, 0.806136, -0.591730)), L, G, Gp, max_order, numVerts);
    float9 w34 = evalLight((float3(-0.553127, 0.319344, 0.769461)), L, G, Gp, max_order, numVerts);
    float9 w35 = evalLight((float3(0.707105, 0.707108, -0.000004)), L, G, Gp, max_order, numVerts);
    float9 w36 = evalLight((float3(0.925820, 0.000000, 0.377964)), L, G, Gp, max_order, numVerts);

    // Stage5: Get Light SH Coeff from ZH Coeff
    // -----------------------------------
    float Lcoeff[81];
    for (int i = 0; i < 81; i++) {
        Lcoeff[i] = 0;
    }

    // Light coeffs are linear combination of Directed ZH
    // band-0
    Lcoeff[0] = 0.282095 * SA;

    // band-1
	Lcoeff[1] = dot(float3(2.1995339, 2.50785367, 1.56572711), float3(w20.array[1], w21.array[1], w22.array[1]));
	Lcoeff[2] = dot(float2(-1.82572523, -2.08165037), float2(w20.array[1], w21.array[1]));
	Lcoeff[3] = dot(float3(2.42459869, 1.44790525, 0.90397552), float3(w20.array[1], w21.array[1], w22.array[1]));

    // band-2 / 5-SH
	Lcoeff[4] = dot(float3(-1.33331385, -0.66666684, -0.99999606), float3(w20.array[2], w23.array[2], w24.array[2]));
	Lcoeff[5] = dot(float3(1.1747938, -0.47923799, -0.69556433), float3(w22.array[2], w23.array[2], w24.array[2]));
	Lcoeff[6] = w24.array[2];
	Lcoeff[7] = dot(float3(-1.21710396, 1.58226094, 0.67825711), float3(w20.array[2], w21.array[2], w22.array[2]));
	Lcoeff[7] += dot(float2(-0.27666329, -0.76671491), float2(w23.array[2], w24.array[2]));
	Lcoeff[8] = dot(float2(-1.15470843, -0.57735948), float2(w23.array[2], w24.array[2]));

    // band-3 / 7-SH
    Lcoeff[9] += dot(float3(-0.418128476395, 1.04704832111, 0.418135743058), float3(w22.array[3], w23.array[3], w25.array[3]));
    Lcoeff[10] += dot(float3(-0.217803921828, 1.61365275071, -0.0430709310435), float3(w20.array[3], w21.array[3], w22.array[3]));
    Lcoeff[10] += dot(float3(-1.08141635635, 0.730013109257, -0.906789272616), float3(w23.array[3], w24.array[3], w25.array[3]));
    Lcoeff[11] += dot(float3(0.539792926181, 0.281276817357, -0.53979650602), float3(w22.array[3], w23.array[3], w25.array[3]));
    Lcoeff[12] += -1.00000000026 * w24.array[3];
    Lcoeff[13] += dot(float4(-1.88563738164, 0.934959388519, -1.39846078802, -0.934977410564), float4(w20.array[3], w22.array[3], w23.array[3], w25.array[3]));
    Lcoeff[14] += dot(float3(-0.822588107798, 0.0250955547337, -0.822583092847), float3(w22.array[3], w24.array[3], w25.array[3]));
    Lcoeff[15] += dot(float3(-1.14577301943, 1.03584677217, -0.849735800355), float3(w20.array[3], w22.array[3], w23.array[3]));
    Lcoeff[15] += dot(float3(-0.438905584229, -0.100364975081, -1.36852983602), float3(w24.array[3], w25.array[3], w26.array[3]));
    
    // band-4 / 9-SH
    Lcoeff[16] += dot(float3(-0.694140591095, -1.46594132085, -3.76291455607), float3(w20.array[4], w21.array[4], w22.array[4]));
    Lcoeff[16] += dot(float3(-4.19771773174, -4.41452625915, -5.21937739623), float3(w23.array[4], w24.array[4], w25.array[4]));
    Lcoeff[16] += dot(float3(30.1096083902, -0.582891410482, -25.58700736), float3(w26.array[4], w27.array[4], w28.array[4]));
    Lcoeff[17] += dot(float4(-0.776237001754, -0.497694700099, 0.155804529921, 0.255292423057), float4(w22.array[4], w23.array[4], w24.array[4], w25.array[4]));
    Lcoeff[17] += dot(float3(-0.00123151211175, 0.86352262597, 0.00106337156796), float3(w26.array[4], w27.array[4], w28.array[4]));
    Lcoeff[18] += dot(float3(1.14732747049, -1.93927453351, -4.97819284362), float3(w20.array[4], w21.array[4], w22.array[4]));
    Lcoeff[18] += dot(float3(-4.52057526927, -7.00211058681, -6.90497275343), float3(w23.array[4], w24.array[4], w25.array[4]));
    Lcoeff[18] += dot(float3(39.8336896922, -0.771083185249, -33.8504871326), float3(w26.array[4], w27.array[4], w28.array[4]));
    Lcoeff[19] += dot(float3(0.392392485498, -0.469375435363, 0.146862690526), float3(w22.array[4], w23.array[4], w24.array[4]));
    Lcoeff[19] += dot(float2(-0.883760925422, 0.81431736181), float2(w25.array[4], w27.array[4]));
    Lcoeff[20] += dot(float3(1.00015572278, -0.00110374505123, 0.000937958411459), float3(w24.array[4], w26.array[4], w28.array[4]));
    Lcoeff[21] += dot(float3(7.51111593422, 6.56318513992, 7.31626822687), float3(w22.array[4], w23.array[4], w24.array[4]));
    Lcoeff[21] += dot(float3(7.51109857163, -51.4260730066, 43.7016908482), float3(w25.array[4], w26.array[4], w28.array[4]));
    Lcoeff[22] += dot(float4(-0.61727564343, 0.205352092062, -0.461764665742, -0.617286413191), float4(w22.array[4], w23.array[4], w24.array[4], w25.array[4]));
    Lcoeff[23] += dot(float3(6.71336600734, 5.24419547627, 7.13550000457), float3(w22.array[4], w23.array[4], w24.array[4]));
    Lcoeff[23] += dot(float3(6.71337558899, -51.8339912003, 45.9921960339), float3(w25.array[4], w26.array[4], w28.array[4]));
    Lcoeff[24] += dot(float3(0.466450172383, 1.19684418958, -0.158210638771), float3(w22.array[4], w23.array[4], w24.array[4]));
    Lcoeff[24] += dot(float2(0.466416144347, 0.000906975300098), float2(w25.array[4], w26.array[4]));

    // band-5 / 11-SH
    Lcoeff[25] += dot(float3(0.133023249281, -0.760308430874, -0.132834964007), float3(w22.array[5], w23.array[5], w25.array[5]));
    Lcoeff[25] += dot(float2(0.518450012982, -0.518506289002), float2(w27.array[5], w29.array[5]));
    Lcoeff[26] += dot(float4(-5.14050071652, 1.83087149155, 1.78238433161, 2.02177436206), float4(w20.array[5], w21.array[5], w22.array[5], w23.array[5]));
    Lcoeff[26] += dot(float4(-6.37830163415, 7.15050958772, -11.4689716172, -1.81357297097), float4(w24.array[5], w25.array[5], w26.array[5], w27.array[5]));
    Lcoeff[26] += dot(float3(18.797061252, -0.985019094169, 0.739867930848), float3(w28.array[5], w29.array[5], w30.array[5]));
    Lcoeff[27] += dot(float3(-0.805270111174, -0.127526187772, 0.805305675495), float3(w22.array[5], w23.array[5], w25.array[5]));
    Lcoeff[27] += dot(float2(0.0355293099825, -0.0355325587066), float2(w27.array[5], w29.array[5]));
    Lcoeff[28] += dot(float4(-3.33455180799, 0.473764461749, 1.29183304086, -4.33925590117), float4(w20.array[5], w22.array[5], w23.array[5], w24.array[5]));
    Lcoeff[28] += dot(float3(5.75047620831, -11.0487024546, -1.39272172093), float3(w25.array[5], w26.array[5], w27.array[5]));
    Lcoeff[28] += dot(float3(15.5105028631, -1.49843129853, 1.60213069355), float3(w28.array[5], w29.array[5], w30.array[5]));
    Lcoeff[29] += dot(float3(0.677072222771, -0.826014233465, -0.67715296313), float3(w22.array[5], w23.array[5], w25.array[5]));
    Lcoeff[29] += dot(float2(-0.789705431563, 0.789736851279), float2(w27.array[5], w29.array[5]));
    Lcoeff[30] += -0.999908682153 * w24.array[5];
    Lcoeff[31] += dot(float3(3.96619174114, -0.19550135503, -1.58642864103), float3(w20.array[5], w22.array[5], w23.array[5]));
    Lcoeff[31] += dot(float3(3.47400073804, -5.20598671543, 5.5486664469), float3(w24.array[5], w25.array[5], w26.array[5]));
    Lcoeff[31] += dot(float3(0.897822456022, -10.1396069573, 1.11892856733), float3(w27.array[5], w28.array[5], w29.array[5]));
    Lcoeff[32] += dot(float3(-0.616617693524, 0.650557423215, -0.616596023515), float3(w22.array[5], w24.array[5], w25.array[5]));
    Lcoeff[32] += dot(float2(-0.277058316558, -0.277048507641), float2(w27.array[5], w29.array[5]));
    Lcoeff[33] += dot(float3(5.0039871462, -1.088320635, -2.00153810596), float3(w20.array[5], w22.array[5], w23.array[5]));
    Lcoeff[33] += dot(float3(5.28613602468, -7.40986154696, 16.2642361284), float3(w24.array[5], w25.array[5], w26.array[5]));
    Lcoeff[33] += dot(float3(1.55989830441, -22.7165927272, 1.83889101022), float3(w27.array[5], w28.array[5], w29.array[5]));
    Lcoeff[34] += dot(float3(0.405328070694, -0.0469925687439, 0.405339520518), float3(w22.array[5], w24.array[5], w25.array[5]));
    Lcoeff[34] += dot(float2(-0.608726965055, -0.608728247749), float2(w27.array[5], w29.array[5]));
    Lcoeff[35] += dot(float3(1.09103970866, -1.17451461211, -0.436442419407), float3(w20.array[5], w22.array[5], w23.array[5]));
    Lcoeff[35] += dot(float3(2.3973180386, -2.55290709278, 3.82908736876), float3(w24.array[5], w25.array[5], w26.array[5]));
    Lcoeff[35] += dot(float3(0.665431405575, -6.99714715269, 0.726274850285), float3(w27.array[5], w28.array[5], w29.array[5]));

    // band-6 / 13-SH
    Lcoeff[36] += dot(float4(110.407306498, 121.297251387, 263.985067401, 110.725812649), float4(w20.array[6], w21.array[6], w22.array[6], w23.array[6]));
    Lcoeff[36] += dot(float3(322.033930881, 162.479577242, 1755.80451307), float3(w24.array[6], w25.array[6], w26.array[6]));
    Lcoeff[36] += dot(float3(-14.7032957811, -1553.88668558, 64.4632133415), float3(w27.array[6], w28.array[6], w29.array[6]));
    Lcoeff[36] += dot(float3(-22.0873701656, -61.3998433111, 187.359713154), float3(w30.array[6], w31.array[6], w32.array[6]));
    Lcoeff[37] += dot(float4(-0.00111818967084, -0.00123422458098, 0.336682549226, -0.00132691821183), float4(w20.array[6], w21.array[6], w22.array[6], w23.array[6]));
    Lcoeff[37] += dot(float4(-0.0034983376219, -0.341167543299, -0.01890204627, -0.723728647079), float4(w24.array[6], w25.array[6], w26.array[6], w27.array[6]));
    Lcoeff[37] += dot(float3(0.0167673427229, 0.723491153889, -0.00198206765087), float3(w28.array[6], w29.array[6], w32.array[6]));
    Lcoeff[38] += dot(float4(15.3227047788, 14.2753461798, 12.4149247001, 1.340434285), float4(w20.array[6], w21.array[6], w22.array[6], w23.array[6]));
    Lcoeff[38] += dot(float3(16.4050796616, 5.20690955596, 93.7391441162), float3(w24.array[6], w25.array[6], w26.array[6]));
    Lcoeff[38] += dot(float3(-5.70151762203, -78.7068450207, 1.01952589216), float3(w27.array[6], w28.array[6], w29.array[6]));
    Lcoeff[38] += dot(float3(-0.0124081990428, -2.71358001541, 13.8096132986), float3(w30.array[6], w31.array[6], w32.array[6]));
    Lcoeff[39] += dot(float4(-0.00117951166106, -1.68851028063, 0.499987655626, 0.0811287014828), float4(w21.array[6], w22.array[6], w23.array[6], w24.array[6]));
    Lcoeff[39] += dot(float4(0.924525864736, -0.0180671469572, -0.53953363757, 0.0160213923983), float4(w25.array[6], w26.array[6], w27.array[6], w28.array[6]));
    Lcoeff[39] += dot(float3(-1.22120890317, 1.93471815616, -0.00189817210815), float3(w29.array[6], w31.array[6], w32.array[6]));
    Lcoeff[40] += dot(float4(14.2349433166, 15.6374252438, 13.8493129894, 2.43692566103), float4(w20.array[6], w21.array[6], w22.array[6], w23.array[6]));
    Lcoeff[40] += dot(float3(16.7738688981, 5.95348061509, 102.676997666), float3(w24.array[6], w25.array[6], w26.array[6]));
    Lcoeff[40] += dot(float3(-4.85746148998, -86.2109081623, 2.50489677852), float3(w27.array[6], w28.array[6], w29.array[6]));
    Lcoeff[40] += dot(float3(-0.0135530622167, -2.97256381466, 15.1268484906), float3(w30.array[6], w31.array[6], w32.array[6]));
    Lcoeff[41] += dot(float4(0.0011471762337, 0.00134013654614, 1.90642287302, -0.949799062166), float4(w20.array[6], w21.array[6], w22.array[6], w23.array[6]));
    Lcoeff[41] += dot(float4(-0.156687609791, -0.460381337914, 0.0194055069076, 0.614031832908), float4(w24.array[6], w25.array[6], w26.array[6], w27.array[6]));
    Lcoeff[41] += dot(float4(-0.017163787015, 2.72661250835, -3.67080744599, 0.00207762117982), float4(w28.array[6], w29.array[6], w31.array[6], w32.array[6]));
    Lcoeff[42] += dot(float3(1.00039252313, 0.00181905245706, -0.00166508494552), float3(w24.array[6], w26.array[6], w28.array[6]));
    Lcoeff[43] += dot(float4(0.0133463019389, 0.0153654992493, 8.78743258451, 5.81522851174), float4(w20.array[6], w21.array[6], w22.array[6], w23.array[6]));
    Lcoeff[43] += dot(float3(12.5550974713, 8.77413125608, 58.4922368204), float3(w24.array[6], w25.array[6], w26.array[6]));
    Lcoeff[43] += dot(float3(1.34231351205, -53.541181195, 1.35225002587), float3(w27.array[6], w28.array[6], w29.array[6]));
    Lcoeff[43] += dot(float3(-0.0032149390263, -0.00792693014645, 3.26902843874), float3(w30.array[6], w31.array[6], w32.array[6]));
    Lcoeff[44] += dot(float3(-0.565026447733, -0.440924426872, -0.803140025769), float3(w22.array[6], w23.array[6], w24.array[6]));
    Lcoeff[44] += dot(float3(-0.565750951282, 0.0106971854579, 0.465219496817), float3(w25.array[6], w26.array[6], w27.array[6]));
    Lcoeff[44] += dot(float3(-0.00942810590012, 0.465803316978, 0.00121182174646), float3(w28.array[6], w29.array[6], w32.array[6]));
    Lcoeff[45] += dot(float4(0.0120591281871, 0.0140081081238, 8.84231008623, 6.30335116571), float4(w20.array[6], w21.array[6], w22.array[6], w23.array[6]));
    Lcoeff[45] += dot(float3(13.3925214762, 8.83016166486, 63.6145973482), float3(w24.array[6], w25.array[6], w26.array[6]));
    Lcoeff[45] += dot(float3(1.11030347967, -59.2232957105, 1.11930309013), float3(w27.array[6], w28.array[6], w29.array[6]));
    Lcoeff[45] += dot(float3(-0.00299575240939, -0.00721839822023, 4.87051865893), float3(w30.array[6], w31.array[6], w32.array[6]));
    Lcoeff[46] += dot(float3(0.651047411165, -0.429335217103, 0.106976359799), float3(w22.array[6], w23.array[6], w24.array[6]));
    Lcoeff[46] += dot(float3(0.650368759846, 0.0109636003151, 0.306691437259), float3(w25.array[6], w26.array[6], w27.array[6]));
    Lcoeff[46] += dot(float3(-0.00965411398733, 0.307235947971, 0.00120365403511), float3(w28.array[6], w29.array[6], w32.array[6]));
    Lcoeff[47] += dot(float4(0.00324439166172, 0.00369123568018, 0.800772124437, 1.63137398167), float4(w20.array[6], w21.array[6], w22.array[6], w23.array[6]));
    Lcoeff[47] += dot(float4(1.43566157126, 0.797501947007, 9.03263537993, 0.535950980597), float4(w24.array[6], w25.array[6], w26.array[6], w27.array[6]));
    Lcoeff[47] += dot(float4(-7.2757158051, 0.538410039638, -0.00198357688491, 1.7022272091), float4(w28.array[6], w29.array[6], w31.array[6], w32.array[6]));
    Lcoeff[48] += dot(float4(0.00156677824614, 0.00170502607019, -0.0939749600966, -0.872048853744), float4(w20.array[6], w21.array[6], w22.array[6], w23.array[6]));
    Lcoeff[48] += dot(float4(0.00440208146779, -0.0953605716075, 0.0240666470029, -0.540590886575), float4(w24.array[6], w25.array[6], w26.array[6], w27.array[6]));
    Lcoeff[48] += dot(float3(-0.0212662429579, -0.539478586759, 0.00260075917574), float3(w28.array[6], w29.array[6], w32.array[6]));

    // band-7 / 15-SH
    Lcoeff[49] += dot(float3(-0.0701805272893, 0.98255354265, 0.0686440478357), float3(w22.array[7], w23.array[7], w25.array[7]));
    Lcoeff[49] += dot(float3(-0.550199211871, 0.550738563391, -0.00124904263303), float3(w27.array[7], w29.array[7], w33.array[7]));
    Lcoeff[50] += dot(float4(-37.3285759367, -5.7563959375, 44.5510845679, 53.5563319108), float4(w20.array[7], w22.array[7], w23.array[7], w24.array[7]));
    Lcoeff[50] += dot(float4(-114.931760993, 26.7620910072, 76.1871627236, 18.7459525504), float4(w25.array[7], w26.array[7], w27.array[7], w28.array[7]));
    Lcoeff[50] += dot(float3(-32.9000483785, -7.89959890218, 45.2203492106), float3(w29.array[7], w30.array[7], w31.array[7]));
    Lcoeff[50] += dot(float3(24.5638783781, -109.38781242, 0.00341946781379), float3(w32.array[7], w33.array[7], w34.array[7]));
    Lcoeff[51] += dot(float3(0.60312464974, 0.514140412839, -0.601853534614), float3(w22.array[7], w23.array[7], w25.array[7]));
    Lcoeff[51] += dot(float2(0.456751969399, -0.457204648754), float2(w27.array[7], w29.array[7]));
    Lcoeff[52] += dot(float4(-5.52955347687, -4.34348722462, 4.3438125575, 7.88948108037), float4(w20.array[7], w22.array[7], w23.array[7], w24.array[7]));
    Lcoeff[52] += dot(float3(-12.6234155916, 6.34843469899, 6.2109893889), float3(w25.array[7], w26.array[7], w27.array[7]));
    Lcoeff[52] += dot(float3(-0.00429379248923, -0.756126464499, 0.424399863695), float3(w28.array[7], w29.array[7], w31.array[7]));
    Lcoeff[52] += dot(float3(3.60062053248, -9.77362846891, -2.02096723974), float3(w32.array[7], w33.array[7], w34.array[7]));
    Lcoeff[53] += dot(float4(-1.84219028026, -0.800561174358, 1.84228914268, -1.76578138065), float4(w22.array[7], w23.array[7], w25.array[7], w27.array[7]));
    Lcoeff[53] += dot(float3(1.76570658332, -2.58447322137, 2.58446953723), float3(w29.array[7], w31.array[7], w33.array[7]));
    Lcoeff[54] += dot(float4(4.28585473464, -5.57083870671, 3.88299131276, -1.21598103809), float4(w20.array[7], w21.array[7], w22.array[7], w23.array[7]));
    Lcoeff[54] += dot(float4(-1.02579013048, 0.421658914765, -6.27175023832, 1.89012154573), float4(w24.array[7], w25.array[7], w26.array[7], w27.array[7]));
    Lcoeff[54] += dot(float3(0.0116877614083, -3.62740434708, -0.00168583412282), float3(w28.array[7], w29.array[7], w30.array[7]));
    Lcoeff[54] += dot(float3(2.90947495533, 2.71420375237, -1.15571591609), float3(w31.array[7], w32.array[7], w33.array[7]));
    Lcoeff[55] += dot(float4(1.33143227971, 1.12171235628, -1.33021306817, 2.17150195324), float4(w22.array[7], w23.array[7], w25.array[7], w27.array[7]));
    Lcoeff[55] += dot(float3(-2.17190603892, 2.68546413814, -2.68475126266), float3(w29.array[7], w31.array[7], w33.array[7]));
    Lcoeff[56] += -0.999785087625 * w24.array[7];
    Lcoeff[57] += dot(float4(-4.98353234343, -0.60335722233, 5.06180432247, 5.983807289), float4(w20.array[7], w22.array[7], w23.array[7], w24.array[7]));
    Lcoeff[57] += dot(float4(-12.4335970683, 5.62509069724, 8.45051745261, -0.00408078300172), float4(w25.array[7], w26.array[7], w27.array[7], w28.array[7]));
    Lcoeff[57] += dot(float4(-2.8882608379, 4.59925710784, 2.29940628779, -11.9972155397), float4(w29.array[7], w31.array[7], w32.array[7], w33.array[7]));
    Lcoeff[58] += dot(float4(-0.87292844039, 0.853086268401, -0.872716127179, 0.548554049149), float4(w22.array[7], w24.array[7], w25.array[7], w27.array[7]));
    Lcoeff[58] += dot(float3(0.548734794611, -1.07309588446, -1.07281880178), float3(w29.array[7], w31.array[7], w33.array[7]));
    Lcoeff[59] += dot(float4(-6.71247285675, -2.43207100005, 6.81762410543, 9.33449029193), float4(w20.array[7], w22.array[7], w23.array[7], w24.array[7]));
    Lcoeff[59] += dot(float3(-18.3657135974, 2.02534151283, 11.8803536762), float3(w25.array[7], w26.array[7], w27.array[7]));
    Lcoeff[59] += dot(float3(6.40090944988, -3.39122648096, 0.00113658729748), float3(w28.array[7], w29.array[7], w30.array[7]));
    Lcoeff[59] += dot(float3(5.38458251565, 4.50642495235, -16.9684952577), float3(w31.array[7], w32.array[7], w33.array[7]));
    Lcoeff[60] += dot(float3(0.8151794721, -0.201027115556, 0.81519502336), float3(w22.array[7], w24.array[7], w25.array[7]));
    Lcoeff[60] += dot(float2(0.181660069418, 0.181680334223), float2(w27.array[7], w29.array[7]));
    Lcoeff[61] += dot(float4(-1.90839389632, -1.76712931026, 1.93832899575, 3.89588368627), float4(w20.array[7], w22.array[7], w23.array[7], w24.array[7]));
    Lcoeff[61] += dot(float4(-6.29727952971, 2.48144221527, 3.24427850885, -1.09768856807), float4(w25.array[7], w26.array[7], w27.array[7], w29.array[7]));
    Lcoeff[61] += dot(float3(0.748671845519, 2.61667793247, -5.60663032021), float3(w31.array[7], w32.array[7], w33.array[7]));
    Lcoeff[62] += dot(float4(-0.270895144417, -0.271153759143, 0.820526195092, 0.820278327508), float4(w22.array[7], w25.array[7], w27.array[7], w29.array[7]));
    Lcoeff[63] += dot(float4(-0.00126785006862, 1.20665045598, 0.00162281551801, -0.963247933527), float4(w20.array[7], w22.array[7], w23.array[7], w24.array[7]));
    Lcoeff[63] += dot(float4(1.20267970448, -1.77208455178, -0.986776039728, 0.0024149973414), float4(w25.array[7], w26.array[7], w27.array[7], w28.array[7]));
    Lcoeff[63] += dot(float4(-0.990721508828, 0.583109464442, 0.451361607718, 0.577458802184), float4(w29.array[7], w31.array[7], w32.array[7], w33.array[7]));

    // band-8 / 17-SH
    Lcoeff[64] += dot(float4(1.53749427566, 0.909584774471, -12.8120323661, 0.431904370785), float4(w20.array[8], w21.array[8], w22.array[8], w23.array[8]));
    Lcoeff[64] += dot(float4(-12.3258435464, -13.1146498411, 88.2060200532, -12.2132842255), float4(w24.array[8], w25.array[8], w26.array[8], w27.array[8]));
    Lcoeff[64] += dot(float3(-111.819986738, -11.6817657491, 0.404047756365), float3(w28.array[8], w29.array[8], w30.array[8]));
    Lcoeff[64] += dot(float3(12.0467189584, -0.994477402231, 12.180903566), float3(w31.array[8], w32.array[8], w33.array[8]));
    Lcoeff[64] += dot(float3(-0.509165463671, 0.214834626621, 32.2725793151), float3(w34.array[8], w35.array[8], w36.array[8]));
    Lcoeff[65] += dot(float4(0.02820380227, -0.0747431249236, 2.92512698977, -0.112167836437), float4(w20.array[8], w21.array[8], w22.array[8], w23.array[8]));
    Lcoeff[65] += dot(float4(2.79794117593, 3.55063984248, -21.5072045956, 3.93193106867), float4(w24.array[8], w25.array[8], w26.array[8], w27.array[8]));
    Lcoeff[65] += dot(float3(27.3991444823, 1.94595833674, 0.0115842799386), float3(w28.array[8], w29.array[8], w30.array[8]));
    Lcoeff[65] += dot(float3(-2.64814758249, 0.42709474181, -2.94245915297), float3(w31.array[8], w32.array[8], w33.array[8]));
    Lcoeff[65] += dot(float3(-0.0686794362676, 0.00655079684495, -7.93317808252), float3(w34.array[8], w35.array[8], w36.array[8]));
    Lcoeff[66] += dot(float4(0.391342450648, -1.02492254371, 125.326506712, -4.6445432842), float4(w20.array[8], w21.array[8], w22.array[8], w23.array[8]));
    Lcoeff[66] += dot(float4(109.33702209, 125.208534258, -851.51589026, 114.502675185), float4(w24.array[8], w25.array[8], w26.array[8], w27.array[8]));
    Lcoeff[66] += dot(float3(1087.75897489, 113.224769946, -1.50797447135), float3(w28.array[8], w29.array[8], w30.array[8]));
    Lcoeff[66] += dot(float3(-108.466362134, 18.4164681113, -109.583142595), float3(w31.array[8], w32.array[8], w33.array[8]));
    Lcoeff[66] += dot(float3(-0.937471108497, 1.39128181347, -316.5905216), float3(w34.array[8], w35.array[8], w36.array[8]));
    Lcoeff[67] += dot(float4(0.119153299569, -0.315481464262, 14.0614420783, -0.473010097461), float4(w20.array[8], w21.array[8], w22.array[8], w23.array[8]));
    Lcoeff[67] += dot(float4(11.7965154114, 13.2422133423, -90.6761509524, 12.8951524656), float4(w24.array[8], w25.array[8], w26.array[8], w27.array[8]));
    Lcoeff[67] += dot(float3(115.516527355, 11.8873201428, 0.049144869583), float3(w28.array[8], w29.array[8], w30.array[8]));
    Lcoeff[67] += dot(float3(-11.1652789635, 1.80037152648, -12.4058801396), float3(w31.array[8], w32.array[8], w33.array[8]));
    Lcoeff[67] += dot(float3(-0.289812743287, 0.0276724657153, -33.4465000918), float3(w34.array[8], w35.array[8], w36.array[8]));
    Lcoeff[68] += dot(float4(-0.0291848617478, 0.245516477549, 118.116347051, -3.38172094147), float4(w20.array[8], w21.array[8], w22.array[8], w23.array[8]));
    Lcoeff[68] += dot(float4(104.366969862, 117.524021629, -813.126174282, 108.739088105), float4(w24.array[8], w25.array[8], w26.array[8], w27.array[8]));
    Lcoeff[68] += dot(float3(1040.02520445, 108.127023707, -1.06230167797), float3(w28.array[8], w29.array[8], w30.array[8]));
    Lcoeff[68] += dot(float3(-102.174880088, 18.9774125832, -103.213055714), float3(w31.array[8], w32.array[8], w33.array[8]));
    Lcoeff[68] += dot(float3(-1.81889823372, -0.555751123944, -303.112465503), float3(w34.array[8], w35.array[8], w36.array[8]));
    Lcoeff[69] += dot(float4(-0.395035183387, 1.04586983477, -45.5572484204, 1.56813129402), float4(w20.array[8], w21.array[8], w22.array[8], w23.array[8]));
    Lcoeff[69] += dot(float4(-39.1074829737, -44.9591485736, 300.607189459, -41.1873913838), float4(w24.array[8], w25.array[8], w26.array[8], w27.array[8]));
    Lcoeff[69] += dot(float3(-382.957356484, -40.9708031646, -0.162930068218), float3(w28.array[8], w29.array[8], w30.array[8]));
    Lcoeff[69] += dot(float3(39.2832393404, -5.96855234135, 38.8592527753), float3(w31.array[8], w32.array[8], w33.array[8]));
    Lcoeff[69] += dot(float3(0.960789726517, -0.0917705056566, 110.880958708), float3(w34.array[8], w35.array[8], w36.array[8]));
    Lcoeff[70] += dot(float4(0.456810706057, -1.19610219984, 145.14169054, -3.60731438449), float4(w20.array[8], w21.array[8], w22.array[8], w23.array[8]));
    Lcoeff[70] += dot(float4(128.010827764, 145.004047784, -993.769284631, 134.16474103), float4(w24.array[8], w25.array[8], w26.array[8], w27.array[8]));
    Lcoeff[70] += dot(float3(1269.47881154, 132.673316481, -1.75982783541), float3(w28.array[8], w29.array[8], w30.array[8]));
    Lcoeff[70] += dot(float3(-125.717600801, 21.4930188871, -127.020961656), float3(w31.array[8], w32.array[8], w33.array[8]));
    Lcoeff[70] += dot(float3(-1.09412783044, -0.92615123861, -369.479614638), float3(w34.array[8], w35.array[8], w36.array[8]));
    Lcoeff[71] += dot(float4(0.367481338147, -0.972917666468, 41.653343463, -1.45876523059), float4(w20.array[8], w21.array[8], w22.array[8], w23.array[8]));
    Lcoeff[71] += dot(float4(36.3807277786, 42.5517382838, -279.647500028, 38.6375231848), float4(w24.array[8], w25.array[8], w26.array[8], w27.array[8]));
    Lcoeff[71] += dot(float3(356.255877922, 37.7922040017, 0.151565737021), float3(w28.array[8], w29.array[8], w30.array[8]));
    Lcoeff[71] += dot(float3(-35.8358023732, 5.55242780488, -36.8581681038), float3(w31.array[8], w32.array[8], w33.array[8]));
    Lcoeff[71] += dot(float3(-0.893804648761, 0.0853291679856, -103.149863518), float3(w34.array[8], w35.array[8], w36.array[8]));
    Lcoeff[72] += dot(float4(0.00188542464872, 1.00165569552, 0.00187041518417, -0.0123197787928), float4(w22.array[8], w24.array[8], w25.array[8], w26.array[8]));
    Lcoeff[72] += dot(float3(0.00169115843328, 0.0156826399574, 0.00170503250921), float3(w27.array[8], w28.array[8], w29.array[8]));
    Lcoeff[72] += dot(float3(-0.00167433376791, -0.00165711826159, -0.00449049345544), float3(w31.array[8], w33.array[8], w36.array[8]));
    Lcoeff[73] += dot(float4(0.00405256436808, -0.00262582264354, 63.550287794, -2.30613108952), float4(w20.array[8], w21.array[8], w22.array[8], w23.array[8]));
    Lcoeff[73] += dot(float4(55.7252304106, 63.5538445478, -418.347495226, 57.5308076948), float4(w24.array[8], w25.array[8], w26.array[8], w27.array[8]));
    Lcoeff[73] += dot(float4(532.655148546, 57.521374147, -0.00352791192438, -55.9572413108), float4(w28.array[8], w29.array[8], w30.array[8], w31.array[8]));
    Lcoeff[73] += dot(float4(7.99539527162, -55.9579600129, 0.00456856106269, -152.990616437), float4(w32.array[8], w33.array[8], w35.array[8], w36.array[8]));
    Lcoeff[74] += dot(float4(-0.881394274788, 0.473165408429, -0.942062797089, -0.881378435578), float4(w22.array[8], w23.array[8], w24.array[8], w25.array[8]));
    Lcoeff[74] += dot(float4(0.0177306070586, -1.01197251544, -0.0226882112384, -1.0120079567), float4(w26.array[8], w27.array[8], w28.array[8], w29.array[8]));
    Lcoeff[74] += dot(float3(1.4069429521, 1.40690203994, 0.00662191089822), float3(w31.array[8], w33.array[8], w36.array[8]));
    Lcoeff[75] += dot(float4(0.00641782135735, -0.00388890880837, 92.7897539783, -2.83263971836), float4(w20.array[8], w21.array[8], w22.array[8], w23.array[8]));
    Lcoeff[75] += dot(float4(81.5477534167, 92.7953548562, -635.681288259, 84.4996736106), float4(w24.array[8], w25.array[8], w26.array[8], w27.array[8]));
    Lcoeff[75] += dot(float4(814.870318064, 84.4851189109, -0.00533844276317, -81.4656461897), float4(w28.array[8], w29.array[8], w30.array[8], w31.array[8]));
    Lcoeff[75] += dot(float4(13.9482979732, -81.4667603321, 0.00699793551893, -238.52207368), float4(w32.array[8], w33.array[8], w35.array[8], w36.array[8]));
    Lcoeff[76] += dot(float4(1.17167184082, -0.106264795054, 0.271482557427, 1.17166744313), float4(w22.array[8], w23.array[8], w24.array[8], w25.array[8]));
    Lcoeff[76] += dot(float4(-0.00997814053713, 0.393888061274, 0.0127991226512, 0.393896718706), float4(w26.array[8], w27.array[8], w28.array[8], w29.array[8]));
    Lcoeff[76] += dot(float3(-0.89395978918, -0.893949593311, -0.00378935185386), float3(w31.array[8], w33.array[8], w36.array[8]));
    Lcoeff[77] += dot(float4(0.0026121874954, -0.00155417390692, 34.9602498365, -0.615998044572), float4(w20.array[8], w21.array[8], w22.array[8], w23.array[8]));
    Lcoeff[77] += dot(float4(30.2173293689, 34.9625301883, -252.286117615, 32.1648727481), float4(w24.array[8], w25.array[8], w26.array[8], w27.array[8]));
    Lcoeff[77] += dot(float4(324.269439477, 32.1590190028, -0.00211835901653, -29.8023247437), float4(w28.array[8], w29.array[8], w30.array[8], w31.array[8]));
    Lcoeff[77] += dot(float4(7.39874675695, -29.8028199408, 0.00279330634013, -96.341082571), float4(w32.array[8], w33.array[8], w35.array[8], w36.array[8]));
    Lcoeff[78] += dot(float4(-0.387527707899, 0.416763243365, -0.0286688282881, -0.387525584935), float4(w22.array[8], w23.array[8], w24.array[8], w25.array[8]));
    Lcoeff[78] += dot(float4(-0.0105262321023, -0.295992700404, 0.0134560889041, -0.296013229931), float4(w26.array[8], w27.array[8], w28.array[8], w29.array[8]));
    Lcoeff[78] += dot(float3(-0.345151726145, -0.345174064069, -0.00393134444984), float3(w31.array[8], w33.array[8], w36.array[8]));
    Lcoeff[79] += dot(float4(4.90103610639, -1.21333206278, 4.02714177428, 4.90129554805), float4(w22.array[8], w23.array[8], w24.array[8], w25.array[8]));
    Lcoeff[79] += dot(float4(-28.4484536271, 3.91036185913, 37.1656352206, 3.90974932967), float4(w26.array[8], w27.array[8], w28.array[8], w29.array[8]));
    Lcoeff[79] += dot(float4(-3.90148035383, 1.47547837629, -3.90145837153, -9.91757330562), float4(w31.array[8], w32.array[8], w33.array[8], w36.array[8]));
    Lcoeff[80] += dot(float4(0.0684332108322, 1.06483675683, -0.00153837235285, 0.0684374426626), float4(w22.array[8], w23.array[8], w24.array[8], w25.array[8]));
    Lcoeff[80] += dot(float4(-0.0166644698234, 0.593587849063, 0.0213264820211, 0.593536664019), float4(w26.array[8], w27.array[8], w28.array[8], w29.array[8]));
    Lcoeff[80] += dot(float3(-0.0440627637216, -0.0441185577727, -0.00620665050047), float3(w31.array[8], w33.array[8], w36.array[8]));

    L_lm[0][0] = Lcoeff[0];
    L_lm[0][1] = Lcoeff[1];
    L_lm[0][2] = Lcoeff[2];
    L_lm[0][3] = Lcoeff[3];
    L_lm[1][0] = Lcoeff[4];
    L_lm[1][1] = Lcoeff[5];
    L_lm[1][2] = Lcoeff[6];
    L_lm[1][3] = Lcoeff[7];
    L_lm[2][0] = Lcoeff[8];
    L_lm[2][1] = Lcoeff[9];
    L_lm[2][2] = Lcoeff[10];
    L_lm[2][3] = Lcoeff[11];
    L_lm[3][0] = Lcoeff[12];
    L_lm[3][1] = Lcoeff[13];
    L_lm[3][2] = Lcoeff[14];
    L_lm[3][3] = Lcoeff[15];

    return L_lm;
}