//
// Created by Max Rosoff on 9/21/2019.
//

#ifndef GRAPHICS_SPHERE_H
#define GRAPHICS_SPHERE_H

#include <iostream>

#include "../RayTracing/Ray.h"

#include "../../Eigen/Eigen"

class Sphere {

public:

    Sphere() = default;
    Sphere(const Sphere &) = default;
    Sphere &operator=(const Sphere &) = default;
    virtual ~Sphere() = default;

    explicit Sphere(const Eigen::Vector3d &, double, const Material &);
    bool intersectionTest(Ray &) const;

    Eigen::Vector3d position;
    double radius = 0;
    
    Material material;
};

std::ostream &operator<<(std::ostream &, const Sphere &);


#endif //GRAPHICS_SPHERE_H
