//
// Created by Max Rosoff on 10/19/2019.
//

#ifndef GRAPHICS_CAMERA_H
#define GRAPHICS_CAMERA_H

#include <iostream>

#include "Ray.h"

class Camera {

public:

    Camera() = default;
    Camera(const Camera &) = default;
    Camera &operator=(const Camera &) = default;
    ~Camera() = default;

    explicit Camera(const Vector &, const Vector &, const Vector &, const std::vector<double> &, double, const std::vector<double> &);
    Ray pixelRay(int row, int col) const;

    Vector eye;
    Vector lookAtPoint;
    Vector upVector;
    std::vector<double> bounds;
    double focalLength = 0;
    std::vector<double> resolution;

    Vector cameraWVector;
    Vector cameraUVector;
    Vector cameraVVector;

private:

    void setUpUVW();
};

std::ostream &operator<<(std::ostream &, const Camera &);


#endif //GRAPHICS_CAMERA_H
