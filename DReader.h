//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_DRIVERREADER_H
#define GRAPHICS_DRIVERREADER_H

#include <string>
#include <vector>

#include <fstream>
#include <sstream>

#include <cerrno>
#include <cstring>

#include <stdexcept>

#include "../Eigen/Eigen/Eigen"

#include "LightSource.h"
#include "Sphere.h"

class DReader {

public:

    DReader() = default;
    DReader &operator<<(const std::string &);

    DReader(const DReader &) = delete;
    DReader &operator=(const DReader &) = delete;
    ~DReader() = default;

    Eigen::Vector3d eye;
    Eigen::Vector3d lookAtPoint;
    Eigen::Vector3d upVector;
    double focalLength = 0;
    Eigen::Vector4d bounds;
    Eigen::Vector2d resolution;
    Eigen::Vector3d ambientLight;
    std::vector<LightSource> lights;
    std::vector<Sphere> spheres;

private:

    void readDriver(const std::string &);
};


#endif //GRAPHICS_DRIVERREADER_H
