//
// Created by Max Rosoff on 10/19/2019.
//

#ifndef GRAPHICS_RAY_H
#define GRAPHICS_RAY_H

#include <string>
#include <vector>
#include <limits>
#include <iostream>

#include "../Matrix/Vector.h"
#include "../SceneItems/Material.h"

class SceneItem;

class Ray {

public:

    Ray() = default;
    Ray(const Ray &) = default;
    Ray &operator=(const Ray &) = default;
    ~Ray() = default;

    __device__ explicit Ray(const Vector &, const Vector);
    __device__ explicit Ray(const Vector &, const Vector, const Vector &);

    Vector point;
    Vector direction;

    Vector closestIntersectionPoint;
    double closestIntersectionDistance = std::numeric_limits<double>::max();

    const SceneItem* hit = nullptr;
    Material material;
    Vector surfaceNormal;
};

__host__ std::ostream &operator<<(std::ostream &, const Ray &);


#endif //GRAPHICS_RAY_H
