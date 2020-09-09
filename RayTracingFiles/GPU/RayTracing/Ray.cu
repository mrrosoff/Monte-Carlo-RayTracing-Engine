//
// Created by Max Rosoff on 10/19/2019.
//

#include "Ray.h"

using namespace std;

__device__ Ray::Ray(const Vector<3> &point, const Vector<3> direction) :

point(point), direction(direction.normalize())

{}

__device__ Ray::Ray(const Vector<3> &point, const Vector<3> direction, const Vector<3> &surfaceNormal) :

point(point), direction(direction.normalize()), surfaceNormal(surfaceNormal)

{}