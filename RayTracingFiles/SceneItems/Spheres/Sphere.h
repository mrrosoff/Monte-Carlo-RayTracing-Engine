//
// Created by Max Rosoff on 9/21/2019.
//

#ifndef GRAPHICS_SPHERE_H
#define GRAPHICS_SPHERE_H

#include <cmath>
#include <iostream>

#include "../SceneItem.h"
#include "../../Matrix/Vector.h"
#include "../../RayTracing/Ray.h"

class Sphere : public SceneItem {

public:

    Sphere() = default;
    Sphere(const Sphere &) = default;
    Sphere &operator=(const Sphere &) = default;
    virtual ~Sphere() = default;

    explicit Sphere(const Vector &, double, const Material &);

    bool intersectionTest(Ray &) const override;
    Ray makeExitRefrationRay(const Ray &, double, double) const override;

    Vector position;
    double radius = 0;
    Material material;
};

std::ostream &operator<<(std::ostream &, const Sphere &);


#endif //GRAPHICS_SPHERE_H
