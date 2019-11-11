//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_TRANSFORM_H
#define GRAPHICS_TRANSFORM_H

#include <string>
#include <vector>
#include <cmath>

#include "../Eigen/Eigen/Eigen"

class Remap {

public:

    explicit Remap(const std::vector<std::string> &);

    Remap &operator=(const Remap &) = delete;
    ~Remap()= default;

    std::string objPath;
    Eigen::MatrixXd transformation;

private:

    Eigen::Matrix4d findRotationMatrix(Eigen::Vector3d &, double theta);
    Eigen::Matrix4d changeCords(Eigen::Vector3d &);
    int findMinIndex(Eigen::Vector3d &rotationVector);

};


#endif //GRAPHICS_TRANSFORM_H
