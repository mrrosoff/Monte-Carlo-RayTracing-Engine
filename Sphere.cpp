//
// Created by Max Rosoff on 9/21/2019.
//

#include "Sphere.h"

Sphere::Sphere(const Eigen::Vector3d &position, const std::vector<Eigen::Vector3d> &colors, double r) :

position(position), Ka(colors[0]), Kd(colors[1]), Ks(colors[2]), Kr(colors[3]), radius(r)

{}