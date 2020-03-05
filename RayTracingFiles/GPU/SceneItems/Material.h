//
// Created by Max Rosoff on 10/19/2019.
//

#ifndef GRAPHICS_MATERIAL_H
#define GRAPHICS_MATERIAL_H

#include <string>

#include "../Matrix/Vector.h"

class Material {

public:

    Material() = default;
    Material(const Material &) = default;
    Material &operator=(const Material &) = default;
    ~Material() = default;

    __host__ explicit Material(const std::string &, const Vector<3> &, int);

    std::string name;
    Vector<3> albedo;

    // Constructor Parameter to Determine Material Property as an Integer

    // 0 -> Lambertian
    // 1 -> Light
    // 2 -> Mirror
    // 3 -> Glass

    bool isLight = false;
    bool isMirror = false;
    bool isGlass = false;
};


#endif //GRAPHICS_MATERIAL_H
