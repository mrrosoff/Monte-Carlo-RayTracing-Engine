//
// Created by Max Rosoff on 10/19/2019.
//

#include "Camera.h"

using namespace std;

Camera::Camera(const Eigen::Vector3d &eye, const Eigen::Vector3d &lookAtPoint, const Eigen::Vector3d &upVector, const Eigen::Vector4d &bounds, const double focalLength, const Eigen::Vector2d &resolution) :

eye(eye), lookAtPoint(lookAtPoint), upVector(upVector), bounds(bounds), focalLength(focalLength), resolution(resolution)

{
    setUpUVW();
}

void Camera::setUpUVW()
{
    auto makeCameraWVector = eye - lookAtPoint;
    cameraWVector = makeCameraWVector.normalized();

    auto makeCameraUVector = upVector.cross(cameraWVector);
    cameraUVector = makeCameraUVector.normalized();

    cameraVVector = cameraWVector.cross(cameraUVector);
}

Ray Camera::pixelRay(const int row, const int col) const
{
    double left = bounds[0];
    double right = bounds[1];
    double bottom = bounds[2];
    double top = bounds[3];

    int width = resolution[0];
    int height = resolution[1];

    auto xValue = static_cast<double>(row) / (width - 1) * (right - left) + left;
    auto yValue = static_cast<double>(col) / (height - 1) * (bottom - top) + top;

    auto point = eye + (focalLength * cameraWVector) + (xValue * cameraUVector) + (yValue * cameraVVector);
    auto direction = point - eye;
    auto normalizedDirection = direction.normalized();

    return Ray(point, normalizedDirection);
}

ostream &operator<<(ostream &out, const Camera &cam)
{
    Eigen::IOFormat ArrayFormat(Eigen::StreamPrecision, 0, "", ", ", "", "", "[", "]");

    out << "Camera Eye: " << cam.eye.format(ArrayFormat) << '\n';
    out << "Camera Look At: " << cam.lookAtPoint.format(ArrayFormat) << '\n';
    out << "Camera Up: " << cam.upVector.format(ArrayFormat) << '\n';
    out << "Camera Bounds: " << cam.bounds.format(ArrayFormat) << '\n';
    out << "Camera FocalLength: " << cam.focalLength << '\n';
    out << "Camera Resolution: " << cam.resolution.format(ArrayFormat) << '\n';
    out << "Camera W: " << cam.cameraWVector.format(ArrayFormat) << '\n';
    out << "Camera U: " << cam.cameraUVector.format(ArrayFormat) << '\n';
    out << "Camera V: " << cam.cameraVVector.format(ArrayFormat) << '\n';

    return out;
}
