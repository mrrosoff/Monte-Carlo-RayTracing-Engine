//
// Created by Max Rosoff on 10/19/2019.
//

#ifndef GRAPHICS_RAY_H
#define GRAPHICS_RAY_H

#include <string>
#include <vector>
#include <limits>

#include <iostream>

#include "../SceneItems/Material.h"
#include "Eigen"

class Ray {

public:

    Ray() = delete;
    Ray(const Ray &) = default;
    Ray &operator=(const Ray &) = delete;
    ~Ray() = default;

    explicit Ray(const Eigen::Vector3d &, const Eigen::Vector3d &, double = std::numeric_limits<double>::max());

    Eigen::Vector3d point;
    Eigen::Vector3d direction;

    Eigen::Vector3d closestIntersectionPoint;
    double closestIntersectionDistance;
    
    Eigen::Vector3d surfaceNormal;
    Material material;
};

std::ostream &operator<<(std::ostream &, const Ray &);


#endif //GRAPHICS_RAY_H
