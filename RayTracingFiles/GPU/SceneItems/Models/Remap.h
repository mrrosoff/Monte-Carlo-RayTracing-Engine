//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_TRANSFORM_H
#define GRAPHICS_TRANSFORM_H

#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>

#include "../../Matrix/Vector.h"
#include "../../Matrix/Matrix.h"

class Remap {

public:

    Remap() = delete;
    Remap(const Remap &) = default;
    Remap &operator=(const Remap &) = delete;
    ~Remap() = default;

    __host__ explicit Remap(const Vector &, double, const Matrix &, const Matrix &, double smoothingAngle, const std::string &);

    double smoothingAngle = 0;
    std::string objPath;
    Matrix transformation;

private:

    __host__ Matrix findRotationMatrix(const Vector &, double theta) const;
    __host__ Matrix changeCords(const Vector &) const;
    __host__ int findMinIndex(const Vector &rotationVector) const;
};

std::ostream &operator<<(std::ostream &, const Remap &);


#endif //GRAPHICS_TRANSFORM_H
