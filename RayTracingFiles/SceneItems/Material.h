//
// Created by Max Rosoff on 10/19/2019.
//

#ifndef GRAPHICS_MATERIAL_H
#define GRAPHICS_MATERIAL_H

#include <string>
#include <iostream>

#include "Eigen"

class Material {

public:

    Material() = default;
    Material(const Material &) = default;
    Material &operator=(const Material &) = default;
    ~Material() = default;

    explicit Material(const std::string &, const Eigen::Vector3d &, const Eigen::Vector3d &, const Eigen::Vector3d &, const Eigen::Vector3d &, double = 16, double = 1, double = 1, int = 2);

    std::string name;
    Eigen::Vector3d Ka;
    Eigen::Vector3d Kd;
    Eigen::Vector3d Ks;
    Eigen::Vector3d Kr;

    double Ns;
    double Ni;
    double d;
    int illum;
};

std::ostream &operator<<(std::ostream &, const Material &);


#endif //GRAPHICS_MATERIAL_H
