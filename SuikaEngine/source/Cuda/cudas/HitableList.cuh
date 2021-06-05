#ifndef HITABLELISTH
#define HITABLELISTH

class HitableList:public Hitable
{
public:
    __device__ HitableList():Hitable(nullptr){}
    __device__ HitableList(Hitable **l, int n):Hitable(nullptr) {list = l; list_size = n;}
    
    __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
    Hitable **list;
    int list_size;
};

__device__ void CreateHitableList(Hitable** hitable, Hitable **list, int n)
{
    (*hitable) = new HitableList(list, n);
}

__device__ bool HitableList::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
{
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for(int i =0;i<list_size;i++)
    {
        if(list[i]->hit(r,t_min,closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

#endif