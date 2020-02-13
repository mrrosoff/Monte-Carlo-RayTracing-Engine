//
// Created by Max Rosoff on 10/19/2019.
//

#include "Camera.h"

using namespace std;

Camera::Camera(const Vector &eye, const Vector &lookAtPoint, const Vector &upVector, const std::vector<double> &bounds, const double focalLength, const std::vector<double> &resolution) :

eye(eye), lookAtPoint(lookAtPoint), upVector(upVector), bounds(bounds), focalLength(focalLength), resolution(resolution)

{
    setUpUVW();
}

void Camera::setUpUVW()
{
    cameraWVector = (eye - lookAtPoint).normalize();
    cameraUVector = upVector.cross(cameraWVector).normalize();
    cameraVVector = cameraWVector.cross(cameraUVector);
}

Ray Camera::pixelRay(const int row, const int col) const
{
    double left = bounds[0];
    double right = bounds[1];
    double bottom = bounds[2];
    double top = bounds[3];

    int width = static_cast<int>(resolution[0]);
    int height = static_cast<int>(resolution[1]);

    auto xValue = static_cast<double>(row) / (width - 1) * (right - left) + left;
    auto yValue = static_cast<double>(col) / (height - 1) * (bottom - top) + top;

    auto point = eye + (cameraWVector * focalLength) + (cameraUVector * xValue) + (cameraVVector * yValue);
    auto direction = point - eye;
    auto normalizedDirection = direction.normalize();

    return Ray(point, normalizedDirection);
}

ostream &operator<<(ostream &out, const Camera &cam)
{
    out << "Camera Eye: " << cam.eye;
    out << "Camera Look At: " << cam.lookAtPoint;
    out << "Camera Up: " << cam.upVector;
    //out << "Camera Bounds: " << cam.bounds;
    out << "Camera FocalLength: " << cam.focalLength;
    //out << "Camera Resolution: " << cam.resolution;
    out << "Camera W: " << cam.cameraWVector;
    out << "Camera U: " << cam.cameraUVector;
    out << "Camera V: " << cam.cameraVVector;

    return out;
}
