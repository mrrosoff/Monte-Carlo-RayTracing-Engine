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

#include "../../Eigen/Eigen"

class SceneItem;

class Ray {

public:

    Ray() = default;
    Ray(const Ray &) = default;
    Ray &operator=(const Ray &) = default;
    ~Ray() = default;

    explicit Ray(const Eigen::Vector3d &, const Eigen::Vector3d &);
    explicit Ray(const Eigen::Vector3d &, const Eigen::Vector3d &, const Eigen::Vector3d &);

    Eigen::Vector3d point;
    Eigen::Vector3d direction;

    Eigen::Vector3d closestIntersectionPoint;
    double closestIntersectionDistance = std::numeric_limits<double>::max();

    const SceneItem* hit = nullptr;
    Material material;
    Eigen::Vector3d surfaceNormal;
};

std::ostream &operator<<(std::ostream &, const Ray &);


#endif //GRAPHICS_RAY_H
