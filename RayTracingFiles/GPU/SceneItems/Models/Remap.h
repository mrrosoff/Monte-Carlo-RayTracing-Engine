//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_TRANSFORM_H
#define GRAPHICS_TRANSFORM_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

#include "../../Matrix/Vector.h"
#include "../../Matrix/Matrix.h"

class Remap {

public:

    Remap() = delete;
    Remap(const Remap &) = default;
    Remap &operator=(const Remap &) = delete;
    ~Remap() = default;

    __host__ explicit Remap(const Vector<3> &, double, const Matrix<4, 4> &, const Matrix<4, 4> &, double smoothingAngle, const std::string &);

    double smoothingAngle = 0;
    std::string objPath;
    Matrix<4, 4> transformation;

private:

    __host__ Matrix<4, 4> findRotationMatrix(const Vector<3> &, double theta) const;
    __host__ Matrix<4, 4> changeCords(const Vector<3> &) const;
    __host__ int findMinIndex(const Vector<3> &rotationVector) const;
};


#endif //GRAPHICS_TRANSFORM_H
