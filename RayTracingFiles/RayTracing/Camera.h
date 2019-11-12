//
// Created by Max Rosoff on 10/19/2019.
//

#ifndef GRAPHICS_CAMERA_H
#define GRAPHICS_CAMERA_H

#include <iostream>

#include "Ray.h"

#include "Eigen"

class Camera {

public:

    Camera() = default;
    Camera(const Camera &) = default;
    Camera &operator=(const Camera &) = default;
    ~Camera() = default;

    explicit Camera(const Eigen::Vector3d &, const Eigen::Vector3d &, const Eigen::Vector3d &, const Eigen::Vector4d &, double, const Eigen::Vector2d &);
    Ray pixelRay( int row, int col);

    Eigen::Vector3d eye;
    Eigen::Vector3d lookAtPoint;
    Eigen::Vector3d upVector;
    Eigen::Vector4d bounds;
    double focalLength = 0;
    Eigen::Vector2d resolution;

    Eigen::Vector3d cameraWVector;
    Eigen::Vector3d cameraUVector;
    Eigen::Vector3d cameraVVector;

private:

    void setUpUVW();
};

std::ostream &operator<<(std::ostream &, const Camera &);


#endif //GRAPHICS_CAMERA_H
