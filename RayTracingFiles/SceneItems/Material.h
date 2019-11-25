//
// Created by Max Rosoff on 10/19/2019.
//

#ifndef GRAPHICS_MATERIAL_H
#define GRAPHICS_MATERIAL_H

#include <string>
#include <iostream>

#include "../../Eigen/Eigen"

class Material {

public:

    Material() = default;
    Material(const Material &) = default;
    Material &operator=(const Material &) = default;
    ~Material() = default;

    explicit Material(const std::string &, const Eigen::Vector3d &, int);

    std::string name;
    Eigen::Vector3d albedo;

    // Constructor Parameter to Determine Material Property as an Integer

    // 0 -> Nothing
    // 1 -> Light
    // 2 -> Mirror
    // 3 -> Glass

    bool isLight = false;
    bool isMirror = false;
    bool isGlass = false;
};

std::ostream &operator<<(std::ostream &, const Material &);


#endif //GRAPHICS_MATERIAL_H
