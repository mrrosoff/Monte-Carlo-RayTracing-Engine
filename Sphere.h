//
// Created by Max Rosoff on 9/21/2019.
//

#ifndef GRAPHICS_SPHERE_H
#define GRAPHICS_SPHERE_H

#include <vector>

#include "../Eigen/Eigen/Eigen"

class Sphere {

public:

    Sphere() = delete;
    explicit Sphere(const Eigen::Vector3d &, const std::vector<Eigen::Vector3d> &, double);

    Sphere(const Sphere &) = default;
    Sphere &operator=(const Sphere &) = delete;
    ~Sphere() = default;

    Eigen::Vector3d position;
    double radius;
    Eigen::Vector3d Ka;
    Eigen::Vector3d Kd;
    Eigen::Vector3d Ks;
    Eigen::Vector3d Kr;
};


#endif //GRAPHICS_SPHERE_H
