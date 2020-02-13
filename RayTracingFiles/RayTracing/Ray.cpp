//
// Created by Max Rosoff on 10/19/2019.
//

#include "Ray.h"

using namespace std;

Ray::Ray(const Vector &point, const Vector direction) :

point(point), direction(direction.normalize())

{}

Ray::Ray(const Vector &point, const Vector direction, const Vector &surfaceNormal) :

point(point), direction(direction.normalize()), surfaceNormal(surfaceNormal)

{}

ostream &operator<<(ostream &out, const Ray &ray)
{
    out << "Ray Point: \n" << ray.point;
    out << "Ray Direction: \n" << ray.direction;

    return out;
}
