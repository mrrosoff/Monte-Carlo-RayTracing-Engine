//
// Created by Max Rosoff on 10/19/2019.
//

#ifndef GRAPHICS_CAMERA_H
#define GRAPHICS_CAMERA_H

#include "Ray.h"

class Camera {

public:

    Camera() = default;
    Camera(const Camera &) = default;
    Camera &operator=(const Camera &) = default;
    ~Camera() = default;

    __host__ explicit Camera(const Vector<3> &, const Vector<3> &, const Vector<3> &, const Vector<4> &, double, const Vector<2> &);
    __device__ Ray pixelRay(int row, int col) const;

    Vector<3> eye;
    Vector<3> lookAtPoint;
    Vector<3> upVector;
    Vector<4> bounds;
    double focalLength = 0;
    Vector<2> resolution;

    Vector<3> cameraWVector;
    Vector<3> cameraUVector;
    Vector<3> cameraVVector;

private:

    void setUpUVW();
};


#endif //GRAPHICS_CAMERA_H
