//
// Created by Max Rosoff on 10/19/2019.
//

#include "Ray.h"

using namespace std;

Ray::Ray(const Eigen::Vector3d &point, const Eigen::Vector3d &direction, const double distance) :

point(point), direction(direction.normalized()), closestIntersectionDistance(distance)

{}

ostream &operator<<(ostream &out, const Ray &ray)
{
    Eigen::IOFormat ArrayFormat(Eigen::StreamPrecision, 0, "", ", ", "", "", "[", "]");

    out << "Ray Point: \n" << ray.point.format(ArrayFormat) << '\n';
    out << "Ray Direction: \n" << ray.direction.format(ArrayFormat) << '\n';

    return out;
}
