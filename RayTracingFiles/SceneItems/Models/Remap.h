//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_TRANSFORM_H
#define GRAPHICS_TRANSFORM_H

#include <string>
#include <vector>
#include <cmath>
#include <iostream>

#include "../../../Eigen/Eigen"

class Remap {

public:

    Remap() = delete;
    Remap(const Remap &) = default;
    Remap &operator=(const Remap &) = delete;
    ~Remap() = default;

    explicit Remap(const Eigen::Vector3d &, double, const Eigen::Matrix4d &, const Eigen::Matrix4d &, double smoothingAngle, const std::string &);

    double smoothingAngle = 0;
    std::string objPath;
    Eigen::MatrixXd transformation;

private:

    Eigen::Matrix4d findRotationMatrix(const Eigen::Vector3d &, double theta) const;
    Eigen::Matrix4d changeCords(const Eigen::Vector3d &) const;
    int findMinIndex(const Eigen::Vector3d &rotationVector) const;
};

std::ostream &operator<<(std::ostream &, const Remap &);


#endif //GRAPHICS_TRANSFORM_H
