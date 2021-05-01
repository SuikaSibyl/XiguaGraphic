// Written by James McLaren. 10/2016
// Adapted from the code snippets in the wonderful book Ray Tracing In One Weekend by Peter Shirley
// http://in1weekend.blogspot.jp/2016/01/ray-tracing-in-one-weekend.html
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// https://www.shadertoy.com/view/4tV3Rc

// It really did only take 2 days to get this functioning (then one more to add some different scenes and tidy)
// Can't recoment reading the book enough.

// There is a debug mode that can be enabled by defining DEBUG_RAY down below.
// Scene can be fixed by defining FIX_SCENE
// Also change kNumSamplesSlow and kNumSamplesFast if this doesn't run fast enough, or you want uber quality.

// I've been using HLSL too long, I can't go back!
#define float2 vec2
#define float3 vec3
#define float4 vec4
#define MAXFLOAT 65535.0
#define PI 3.14159265

// Material types
#define kLambertian 1
#define kMetal 2
#define kDielectric 3

// Our scenes.
#define kSceneSimple 0
#define kSceneThinGlass 1
#define kSceneComplex 2


// If things run too slow for you, try reducing these numbers
#define kNumSamplesSlow 20
#define kNumSamplesFast 2

//#define USE_RADICAL_INVERSE

// Use these if you want to play with the debug mode
//#define DEBUG_RAY
//#define FIX_SCENE kSceneSimple

// Turn this on to force the debug ray to come in from the left hand side
//#define FORCE_DEBUG_RAY_FROM_THE_SIDE


int g_Scene = kSceneComplex;
float g_Animate = 0.0;

//------------
// Random(ish)
//------------

// https://www.shadertoy.com/view/4l2SRW
float FoldedRadicalInverse(int n, int base)
{
	float inv_base = 1.0 / float(base);
	float inv_base_i = inv_base;
	float val = 0.0;
	int offset = 0;

	for (int i = 0; i < 8; ++i)
	{
		int div = (n + offset) / base;
		int digit = (n + offset) - div * base;
		val += float(digit) * inv_base_i;
		inv_base_i *= inv_base;
		n /= base;
		offset++;
	}

	return val;
}


//Stolen from IQ's "Pool" shader https://www.shadertoy.com/view/4sXGDs#
highp float hash(highp float seed)
{
    return fract(sin(seed)*43758.5453 );
}

float fragRandOffset( float2 fragCoord , float a)
{
	return (hash( dot( fragCoord, vec2(12.9898, 78.233) ) + 1113.1*a ));
}

#ifdef USE_RADICAL_INVERSE
int g_CurrentRandIndex;
float g_FragRandOffset;

void init_rand( float2 fragCoord )
{
    fragCoord += fract(float2(float(iTime*383.0),(iTime*787.0))/953.0)*953.0;
	g_FragRandOffset = fragRandOffset(fragCoord,1.0);
    g_CurrentRandIndex = int(fragRandOffset(fragCoord,2.0)*16.0);

}

float rand()
{
    g_CurrentRandIndex++;
    int idx = g_CurrentRandIndex/4;
    int base = (g_CurrentRandIndex - idx*4)+2;
	return fract(FoldedRadicalInverse(idx,base)+g_FragRandOffset);
}
#else

// different version that doesn't use FoldedRadicalInverse
// Attempting to do a Combined_Linear_Congruential_Generator
// https://en.wikipedia.org/wiki/Combined_Linear_Congruential_Generator
// It's quite possible I have the math wrong/dodgy with regards to using mod.

highp int g_CurrentRand1;
highp int g_CurrentRand2;

//These numbers taken from here https://en.wikipedia.org/wiki/Linear_congruential_generator
#define kMultiplier1 1140671485 
#define kMultiplier2 65793  
    
#define kIncrement1	12820163 
#define kIncrement2 4282663

#define kModulo1 16777216
#define kModulo2 8388608

void init_rand( float2 fragCoord )
{
	fragCoord += fract(float2(float(iTime*383.0),(iTime*787.0))/953.0)*953.0;
	g_CurrentRand1 = int(fragRandOffset(fragCoord,1.0)*float(kMultiplier1));
    g_CurrentRand2 = int(fragRandOffset(fragCoord,1.0)*float(kMultiplier2));
}


float rand()
{
    highp int mul1 = kMultiplier1;
    highp int mul2 = kMultiplier2;
	highp int inc1 = kIncrement1;
    highp int inc2 = kIncrement2;
    highp float mod1 = float(kModulo1);
    highp float mod2 = float(kModulo2); 
        
    // move both internal generators on to their next number
	g_CurrentRand1 = int(mod(float(g_CurrentRand1*mul1 + inc1),mod1));
    g_CurrentRand2 = int(mod(float(g_CurrentRand2*mul2 + inc2),mod2));
    
    // combine them to get something that is hopefully more random
    return fract(float(g_CurrentRand1 - g_CurrentRand2)/mod1);
}


#endif

// Do this from random vars, rather than via the rejection method, because GLSL ES...
float3 random_in_unit_sphere()
{	
    float phi = (rand()*2.0-1.0)*PI;
    float theta = (rand()*2.0-1.0)*PI;
    float costheta = cos(theta);
    float sintheta = sin(theta);
	// float costheta = rand()*2.0-1.0;
	float u = rand();
    
	//float theta = acos( costheta );
    // float sintheta = sqrt(1.0-costheta*costheta);
	float r = pow( u ,1.0/3.0);
    
    float x = r * sintheta * cos( phi );
	float y = r * sintheta * sin( phi );
	float z = r * costheta;
    
    return float3(x,y,z);
}

//----------
// Materials
//----------

struct Material
{
	int type;
    float3 albedo;
    float f;
    float ref_idx;
};

Material materialConstructLambertian(float3 albedo)
{
	Material mat;
    mat.type = kLambertian;
    mat.albedo = albedo;
    mat.f = 0.0;
    mat.ref_idx = 0.0;
    return mat;
}

Material materialConstructMetal(float3 albedo,float fuzz)
{
	Material mat;
    mat.type = kMetal;
    mat.albedo = albedo;
    mat.f = fuzz;
    mat.ref_idx = 0.0;
    return mat;
}

Material materialConstructDielectric(float ri)
{
	Material mat;
    mat.type = kDielectric;
    mat.albedo = float3(1.0);
    mat.f = 0.0;
    mat.ref_idx = ri;
    return mat;
}


//------------
// HitRecord 
//------------

struct HitRecord
{
	float t;
    float3 p;
    float3 normal;
    Material material;
};

//------
// Ray
//------
struct Ray
{
	float3 o;
    float3 d;
};

Ray rayConstruct(in float3 origin,in float3 direction) 
{
    Ray r;
    r.o = origin;
    r.d = direction;
    return r;
}    
    
float3 rayPointAtParameter(in Ray r,float t)
{
	return r.o + t*r.d;
}

//--------
// Sphere
//--------
struct Sphere
{
	float3 center;
    float radius;
};

Sphere sphereConstruct(in float3 center,in float radius)
{
	Sphere s;
    s.center = center;
    s.radius = radius;
    return s;
}

bool sphereHit(in Sphere s, in Ray r,float t_min,float t_max,out HitRecord rec)
{
	float3 oc = r.o - s.center;
    float a = dot(r.d,r.d);
    float b = dot(oc, r.d);
    float c = dot(oc,oc) - s.radius*s.radius;
    float discriminant = b*b -a*c;
    if(discriminant > 0.0)
    {
        float temp = (-b-sqrt(discriminant) )/ (a);
        if( (temp < t_max) && (temp > t_min) )
        {
        	rec.t = temp;
            rec.p = rayPointAtParameter(r,rec.t);
            rec.normal =( rec.p-s.center) / s.radius;
            return true;
        }
        temp = (-b+sqrt(discriminant) )/ (a);
        if( (temp < t_max) && (temp > t_min) )
        {
            rec.t = temp;
            rec.p = rayPointAtParameter(r,rec.t);
            rec.normal =( rec.p-s.center) / s.radius;
            return true;
        }
    }

    return false;

}

//--------
// Camera
//--------
struct Camera
{
	float3 origin;
    float3 lower_left_corner;
    float3 horizontal;
    float3 vertical;
    float3 u,v,w;
    float lens_radius;
};


// I don't do rejection sampling like the book as this wouldn't play nice with OpenGL ES.
// Instead, I do Uniform Spherical Sampling using Archimedes' theorem 
// http://repository.upenn.edu/cgi/viewcontent.cgi?article=1188&context=cis_reports
float3 random_in_unit_disk()
{
    float theta = rand()*2.0*PI;
    float r = sqrt(rand());
  
	return float3(r*sin(theta),r*cos(theta),0.0);
}


    
// vfov is top to bottom degrees
Camera cameraConstruct(float3 lookfrom, float3 lookat,float3 vup,float vfov,float aspect,float aperture,float focus_dist)
{
    Camera cam;

    cam.lens_radius = aperture / 2.0;
    float theta = vfov*PI/180.0;
    float half_height = tan(theta/2.0);
    float half_width = aspect * half_height;
    cam.origin = lookfrom;
    cam.w = normalize(lookfrom - lookat);
    cam.u = normalize(cross(vup, cam.w));
    cam.v = cross(cam.w, cam.u);
    cam.lower_left_corner = cam.origin - half_width*focus_dist*cam.u - half_height*focus_dist*cam.v -focus_dist*cam.w;
    cam.horizontal = 2.0*half_width*focus_dist*cam.u;
    cam.vertical = 2.0*half_height*focus_dist*cam.v; 
        
    return cam;
}

Ray cameraGetRay(in Camera cam,in float2 uv)
{
    float3 rd = cam.lens_radius*random_in_unit_disk();
    float3 offset = cam.u * rd.x + cam.v * rd.y;
	return Ray(cam.origin + offset,cam.lower_left_corner + uv.x*cam.horizontal + uv.y*cam.vertical - cam.origin - offset);
}

//------------
// Scattering
//------------

bool refract_me(in float3 v,in float3 n,in float ni_over_nt,out float3 refracted)
{
	float3 uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1.0-
                                                      dt*dt);
    if(discriminant > 0.0)
    {
    	refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);     
        return true;
    }
    else
    {
    	return false;
    }
}

float schlick(float cosine, float ref_idx)
{
	float r0 = (1.0-ref_idx) / (1.0 + ref_idx);
    r0 = r0*r0;
    return r0 + (1.0-r0)*pow((1.0 - cosine),5.0);
}


// Note, the material is held in the hit record so we don't need to pass it as the first arg
bool scatter(in Ray r_in,in HitRecord rec, out float3 attenuation, out Ray scattered)
{
    // No virtual functions in shader land (thank goodness)
    // so test against the type of the material to determine how it should scatter. 
	if(rec.material.type==kLambertian)
    {
    	float3 target = rec.p + rec.normal + random_in_unit_sphere();
        scattered = rayConstruct(rec.p,target-rec.p);
        attenuation = rec.material.albedo;
        return true;
    }
    if(rec.material.type==kMetal)
    {
    	float3 reflected = reflect(normalize(r_in.d),rec.normal);
        scattered = rayConstruct(rec.p,reflected + rec.material.f*random_in_unit_sphere());
        attenuation = rec.material.albedo;
        return (dot(scattered.d,rec.normal)>0.0);
    }
    if(rec.material.type==kDielectric)
    {
        float3 outward_normal;
        float3 reflected = reflect(r_in.d,rec.normal);
        float ni_over_nt;
        attenuation = float3(1.0);
        float3 refracted;
        float reflect_prob;
        float cosine;
        if(dot(r_in.d,rec.normal) > 0.0) 
        {
        	outward_normal = -rec.normal;
            ni_over_nt = rec.material.ref_idx;
            cosine = rec.material.ref_idx * dot(r_in.d,rec.normal)/ length(r_in.d);
        }
        else
        {
        	outward_normal = rec.normal;
            ni_over_nt = 1.0 / rec.material.ref_idx;
            cosine = -dot(r_in.d,rec.normal)/ length(r_in.d);
          
        }
        if(refract_me(r_in.d,outward_normal,ni_over_nt,refracted))
        {
        	
            reflect_prob = schlick(cosine, rec.material.ref_idx); 
        }
        else
        {
            reflect_prob =1.0;
        }
        
        if(rand() < reflect_prob)
        {
        	scattered = rayConstruct(rec.p,reflected);
        }
        else
        {
            scattered = rayConstruct(rec.p,refracted);
        }
        
        return true;
    }
    // what is this material???
    attenuation = float3(0,1,0);
    return true;
}

//---------------
// Hitables...
//--------------- 
// lists etc don't really play nice with shaders, certainly not OpenGL ES....
// So for our hitable list is a procedurally defined (Several different versions here for different scenes)
// The scenes try for the most part to be faithful to the book, with a little bit of embelishment..

// Simple scene that is similar to the ones from chapters 8 and 9 in the book
bool hitablelistHitSimple(in Ray r,in float t_min,in float t_max,out HitRecord rec)
{
	HitRecord temp_rec;
    
    bool hit_anything = false;
    float closest_so_far = t_max;
    
    const int numSpheres=4;
    Sphere spheres[numSpheres]; 
    Material sphere_mats[numSpheres];
    
    // init them
    spheres[0] = sphereConstruct(float3(0.0,0.0,-1.0),0.5);
    spheres[1] = sphereConstruct(float3(0.0,-100.5,-1.0),100.0);
    spheres[2] = sphereConstruct(float3(1.0,0.0,-1.0),0.5);
    spheres[3] = sphereConstruct(float3(-1.0,0.0,-1.0),0.5);

	
 	if(g_Animate>0.2)
    {
    	sphere_mats[0] = materialConstructLambertian(float3(0.1,0.2,0.5));
    }
    else
    {
    	sphere_mats[0] = materialConstructLambertian(float3(0.8,0.3,0.3));
    }
    if(g_Animate>0.6)
    {
    	sphere_mats[1] = materialConstructLambertian(float3(0.8,0.3,0.0));
    }
    else
    {
    	sphere_mats[1] = materialConstructLambertian(float3(0.8,0.8,0.0));
    }
    if(g_Animate>0.8)
    {
        sphere_mats[2] = materialConstructMetal(float3(0.8,0.6,0.2),mix(0.3,0.95,smoothstep(0.8,0.95,g_Animate)));
    }
    else
    {
		sphere_mats[2] = materialConstructMetal(float3(0.8,0.6,0.2),0.3);
    }
    if(g_Animate>0.4)
    {
        if(g_Animate>0.8)
        {
    		sphere_mats[3] = materialConstructDielectric(mix(1.5,2.67,smoothstep(0.8,0.95,g_Animate)));
        }
        else
        {
        	sphere_mats[3] = materialConstructDielectric(1.5);
        }
    }
    else
    {
    	sphere_mats[3] = materialConstructMetal(float3(0.8,0.8,0.8),0.8);
    }
    
    // one loop per type of thing
    for(int i=0;i<numSpheres;++i)
    {
    	if(sphereHit(spheres[i],r,t_min,closest_so_far, temp_rec))
        {
        	hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
            rec.material = sphere_mats[i];
        }
    }
    
    return hit_anything;
}


// Thin glass scene from the end of chapter 9
bool hitablelistHitThinGlass(in Ray r,in float t_min,in float t_max,out HitRecord rec)
{
	HitRecord temp_rec;
    
    bool hit_anything = false;
    float closest_so_far = t_max;
    
    const int numSpheres=5;
    Sphere spheres[numSpheres]; 
    Material sphere_mats[numSpheres];
    
    // init them
    spheres[0] = sphereConstruct(float3(0.0,0.0,-1.0),0.5);
    spheres[1] = sphereConstruct(float3(0.0,-100.5,-1.0),100.0);
    spheres[2] = sphereConstruct(float3(1.0,0.0,-1.0),0.5);
    spheres[3] = sphereConstruct(float3(-1.0,0.0,-1.0),0.5);
    spheres[4] = sphereConstruct(float3(-1.0,0.0,-1.0),-0.45);
	sphere_mats[0] = materialConstructLambertian(float3(0.1,0.2,0.5));
    sphere_mats[1] = materialConstructLambertian(float3(0.8,0.8,0.0));
    sphere_mats[2] = materialConstructMetal(float3(0.8,0.6,0.2),0.3);
    sphere_mats[3] = materialConstructDielectric(1.5);
    sphere_mats[4] = materialConstructDielectric(1.5);
    
    // one loop per type of thing
    for(int i=0;i<numSpheres;++i)
    {
    	if(sphereHit(spheres[i],r,t_min,closest_so_far, temp_rec))
        {
        	hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
            rec.material = sphere_mats[i];
        }
    }
    
    return hit_anything;
}

// Complex scene similar to the one in chapter 12 (though slightly cut down)
bool hitablelistHitComplex(in Ray r,in float t_min,in float t_max,out HitRecord rec)
{
	HitRecord temp_rec;
    
    bool hit_anything = false;
    float closest_so_far = t_max;
    
    const int numSpheres=4;
    Sphere spheres[numSpheres]; 
    Material sphere_mats[numSpheres];
    
    // init them
    spheres[0] = sphereConstruct(float3(0.0,-400.0,0.0),400.0);
    spheres[1] = sphereConstruct(float3(0.0,1.0,0.0),1.0);
    spheres[2] = sphereConstruct(float3(-4.0,1.0,0.0),1.0);
    spheres[3] = sphereConstruct(float3(4.0,1.0,0.0),1.0);
	sphere_mats[0] = materialConstructLambertian(float3(0.5,0.5,0.5));
    sphere_mats[1] = materialConstructDielectric(1.5);
    sphere_mats[2] = materialConstructLambertian(float3(0.4,0.2,0.1));
    sphere_mats[3] = materialConstructMetal(float3(0.7,0.6,0.5),0.0);
    
    // one loop per type of thing
    for(int i=0;i<numSpheres;++i)
    {
    	if(sphereHit(spheres[i],r,t_min,closest_so_far, temp_rec))
        {
        	hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
            rec.material = sphere_mats[i];
        }
    }
    
    // loop on random spheres....
    
    int rnd=1;
    // a local random that is coherent over all the pixels
#define lrand() hash(float(rnd++))     
    
    // book has more spheres than this, but it gets a bit heavy so this is slightly scaled down
    for(int a = -9; a < 9; a+=2)
    {
    	for(int b = -9; b < 9; b+=2)
        {
            Sphere sphere;
            Material mat;
            
            float choose_mat = hash(float(rnd++));
            float3 center = float3(float(a) + 0.9*lrand(),0.2,float(b)+0.9*lrand());

            
            if(length(center - float3(4.0,0.2,0.0)) > 0.9)
            {
                // add some movement to provide visiual interest...the book doesn't have this :-)
                center.xz += float2(lrand(),lrand())*g_Animate*1.0;
                
                sphere = sphereConstruct(center,0.2);
                if(choose_mat < 0.8)//diffuse
                {
                    mat = materialConstructLambertian(float3(lrand()*lrand(),lrand()*lrand(),lrand()*lrand()));
                }
                else if (choose_mat < 0.95)//metal
                {            	
                    mat = materialConstructMetal(float3(0.5*(1.0+lrand()),0.5*(1.0+lrand()),0.5*(1.0+lrand())),0.5*lrand());
                }
                else //glass
                {
                	mat = materialConstructDielectric(1.5);
                }
                
                if(sphereHit(sphere,r,t_min,closest_so_far, temp_rec))
                {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                    rec.material = mat;
                }              
            }
        }       
    }
#undef lrand
    
    return hit_anything;
}

bool hitablelistHit(in Ray r,in float t_min,in float t_max,out HitRecord rec)
{
	if(g_Scene==kSceneSimple)
    {
    	return hitablelistHitSimple(r,t_min,t_max,rec);
    }
    else if(g_Scene==kSceneThinGlass)
    {
    	return hitablelistHitThinGlass(r,t_min,t_max,rec);
    }
    else
    {
    	return hitablelistHitComplex(r,t_min,t_max,rec);   
    }
}

//---------------
// Color
//---------------

// The color of a ray is?
float3 color(in Ray r)
{
    HitRecord rec;
    
    float3 not_absorbed = float3(1.0);
    
    Ray current_ray = r;
    
    const int max_depth = 8;
    
    for(int i=0;i<max_depth;++i)
    {
        if(hitablelistHit(current_ray,0.00001,MAXFLOAT,rec))
        {
            Ray scattered;
            float3 attenuation;
            
            if(scatter(current_ray,rec,attenuation,scattered))
            {
                current_ray = scattered;
                not_absorbed*=attenuation;
            }
            else
            {
            	not_absorbed=float3(0,0,0);
                break;
            }           
        }
        else
        {
        	break;
        }
    }

    float3 unit_direction = normalize(r.d);
    float t = 0.5*(unit_direction.y + 1.0);
    float3 sky_col = (1.0-t)*float3(1.0,1.0,1.0) + t*float3(0.5,0.7,1.0);
    return not_absorbed*sky_col;
}


//----------------
// Debug!
//----------------
// I ended up needing some rough debugging functions
// because I couldn't work out for the longest time what was wrong with my refractive spheres
// debug_trace is largely a copy of the color() function, but it spits out a debug representation
// of where the "photons" are at each bounce (colours change)
// define DEBUG_RAY and choose a scene with FIX_SCENE for best effect

float3 getDebugDepthCol(int depth)
{
    int depth_plus1=depth;

    // I miss bitwise operations.....
    int temp_001 = (depth_plus1/2);
    float bit_001 = (depth_plus1-(temp_001*2)) != 0 ? 1.0 : 0.0;
    int temp_010 = (depth_plus1/4);
    float bit_010 = ((depth_plus1/2) - (temp_010*2)) != 0 ? 1.0 : 0.0;
    int temp_100 = (depth_plus1/8);
    float bit_100 = ((depth_plus1/4) - (temp_100*2)) != 0 ? 1.0 : 0.0;            

    return float3(bit_001,bit_010,bit_100);
}

void getSegmentDebugCol(inout float4 debug_col,in float3 depth_col, in Ray current_ray, in Ray pixel_ray, in HitRecord rec,float t)
{
    float3 p = rayPointAtParameter(current_ray,rec.t*t);
    float3 dir_ray = normalize(p-pixel_ray.o);
    if(dot(dir_ray,normalize(pixel_ray.d))>0.99990)
    {
        debug_col = max(float4(depth_col,1),debug_col);
    }
}

float4 debug_trace(in Ray r,in Ray pixel_ray)
{
    HitRecord rec;
    
    float3 not_absorbed = float3(1.0);
    
    Ray current_ray = r;
    
    float4 debug_col = float4(0.0);
    
    const int max_depth = 8;
    
    float3 depth_col = float3(1.0,1.0,1.0);
    
    float loop_time = fract(float(iTime*2.0)/float(max_depth))*float(max_depth);
    
    int last_depth = max_depth-1;
    for(int i=0;i<max_depth;++i)
    {
        if(hitablelistHit(current_ray,0.01,MAXFLOAT,rec))
        {
            Ray scattered;
            float3 attenuation;
                 
            depth_col = getDebugDepthCol(i);
            
 
            float segment_time = loop_time - float(i);
            if((segment_time>=0.0)&&(segment_time<1.0))
            {
                getSegmentDebugCol(debug_col,depth_col,current_ray,pixel_ray,rec,segment_time);
            }
            
            if(scatter(current_ray,rec,attenuation,scattered))
            {
                current_ray = scattered;
                not_absorbed*=attenuation;                         
            }
            else
            {
            	not_absorbed=float3(0,0,0);
                last_depth = i;
                break;
            }
        }
        else
        {
            last_depth = i;
        	break;
        }      
    }
      
    depth_col = getDebugDepthCol(last_depth);
    
    float segment_time = loop_time - float(last_depth);
    if((segment_time>=0.0)&&(segment_time<1.0))
    {
        getSegmentDebugCol(debug_col,depth_col,current_ray,pixel_ray,rec,segment_time);
    }
    return debug_col;
}

//------------------------
//Scene calculation
//------------------------
// I provide two versions slow and fast, so that we have some chance of coping with the
// monster many spheres "complex" scene.

void calcSceneSlow(in Camera cam,in vec2 fragCoord,out float3 col,out float4 debug_col)
{
    float nx = iResolution.x;
    float ny = iResolution.y;
    const int ns = kNumSamplesSlow;
    
	col = float3(0.0);
    debug_col = float4(0.0);
#ifndef FORCE_DEBUG_RAY_FROM_THE_SIDE    
    Ray r_debug = cameraGetRay(cam,(iMouse.xy+float2(0.5))/iResolution.xy);
#else
 	Ray r_debug = rayConstruct(float3(-2.0,0.0,-1.0),float3(1.0,-0.252,0.0));   
#endif    
    
    for(int s = 0;s < ns; ++s)
    {
    	float2 uv = float2(fragCoord.x+rand(),fragCoord.y+rand())/float2(nx,ny);
        Ray r = cameraGetRay(cam,uv);
       	

        col += color(r);
#ifdef DEBUG_RAY        
        debug_col += debug_trace(r_debug,r);
#endif      
    }
    col/=float(ns); 
}

void calcSceneFast(in Camera cam,in vec2 fragCoord,out float3 col,out float4 debug_col)
{
    float nx = iResolution.x;
    float ny = iResolution.y;
    const int ns = kNumSamplesFast;
    
	col = float3(0.0);
    debug_col = float4(0.0);
#ifndef FORCE_DEBUG_RAY_FROM_THE_SIDE    
    Ray r_debug = cameraGetRay(cam,(iMouse.xy+float2(0.5))/iResolution.xy);
#else
 	Ray r_debug = rayConstruct(float3(-2.0,0.0,-1.0),float3(1.0,-0.252,0.0));   
#endif    
    
    for(int s = 0;s < ns; ++s)
    {
    	float2 uv = float2(fragCoord.x+rand(),fragCoord.y+rand())/float2(nx,ny);
        Ray r = cameraGetRay(cam,uv);
       	

        col += color(r);
#ifdef DEBUG_RAY        
        debug_col += debug_trace(r_debug,r);
#endif      
    }
    col/=float(ns); 
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float nx = iResolution.x;
    float ny = iResolution.y;
    const int ns = 5;
    
    init_rand(fragCoord);
    
    Camera cam; 
  
    float loop_time = mod(iTime/10.0,3.0);
    int stage = int(loop_time);
    float local_time = loop_time-float(stage);
    
#ifdef FIX_SCENE
    stage = FIX_SCENE;
    local_time = 0.5;
#endif    
    
    int max_step = ns;
            
	float3 col = float3(0.0);
    float4 debug_col = float4(0.0);
    
    float prev_frame_amount = 0.0;
    
    g_Animate = local_time;
    
    if(stage==1)
    {
        float3 lookfrom = float3(3.0-local_time*14.0,3.0-local_time*3.0,2.0-local_time*1.0);
        float3 lookat = float3(0.0,0.0,-1.0);
        float dist_to_focus = length(lookfrom-lookat);
        float aperture = 2.0-local_time*1.9;

        cam = cameraConstruct(lookfrom, lookat, float3(0.0,1.0,0.0), 20.0, nx/ny, aperture, dist_to_focus);
        
    	g_Scene = kSceneThinGlass;
        prev_frame_amount = 0.85;
        calcSceneSlow(cam,fragCoord,col,debug_col);
    }
    else if(stage==0)
    {
    	Camera cam = cameraConstruct(float3(0.0+local_time*0.2,local_time*0.2,local_time*0.5),float3(0.0,0.0,-1.0),float3(0.0,1.0,0.0),90.0,nx/ny,0.1-local_time*0.09,0.9-(0.6*sin(0.5+local_time*50.0)*(1.0-smoothstep(0.0,0.35,local_time))));
        g_Scene = kSceneSimple;
        prev_frame_amount = 0.85;

        calcSceneSlow(cam,fragCoord,col,debug_col);
    }
	else   
    {
        cam = cameraConstruct(float3(10.0-local_time*1.0,2.0,2.5),float3(0.0,0.0,-1.0),float3(0.0,1.0,0.0),30.0,nx/ny,0.1,20.0);
        g_Scene = kSceneComplex;
        prev_frame_amount = mix(0.7,0.85,max(min(local_time*10.0,1.0),0.0));

        calcSceneFast(cam,fragCoord,col,debug_col);
    }
    
    col = mix(col,debug_col.xyz,debug_col.w);
    
    // Do some very simple temporal anitaliasing to combine samples over frames
    // Currently I'm just assuming that camera motion is slow enough that we can just use
    // the same pixel from the last frame.
    float3 last_col = texture( iChannel0, fragCoord/iResolution.xy ).xyz;

    // make sure we always start off with no previous frame
    prev_frame_amount *= smoothstep(0.0,0.005,local_time);
    
    fragColor = float4(mix(col,last_col,prev_frame_amount),1.0);
}