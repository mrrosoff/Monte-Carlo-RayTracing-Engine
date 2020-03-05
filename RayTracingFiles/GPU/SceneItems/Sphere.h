//
// Created by Max Rosoff on 9/21/2019.
//

#ifndef GRAPHICS_SPHERE_H
#define GRAPHICS_SPHERE_H

#include <cmath>

#include "SceneItem.h"
#include "../Matrix/Vector.h"
#include "../RayTracing/Ray.h"

class Sphere : public SceneItem {

public:

    Sphere() = default;
    Sphere(const Sphere &) = default;
    Sphere &operator=(const Sphere &) = default;
    virtual ~Sphere() = default;

    __host__ explicit Sphere(const Vector<3> &, double, const Material &);

    __device__ bool intersectionTest(Ray &) const override;
    __device__ Ray makeExitRefrationRay(const Ray &, double, double) const override;

    Vector<3> position;
    double radius = 0;
    Material material;
};


#endif //GRAPHICS_SPHERE_H
