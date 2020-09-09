//
// Created by Max Rosoff on 9/21/2019.
//

#include "Sphere.h"

using namespace std;

Sphere::Sphere(const Vector<3> &position, const double radius, const Material &material) :

position(position), radius(radius), material(material)

{}

__device__ bool Sphere::intersectionTest(Ray &ray) const
{
    auto cVector = position - ray.point;
    auto vDist = cVector.dot(ray.direction);
    double dDistSqr = pow(radius, 2) - (cVector.dot(cVector) - pow(vDist, 2));

    if(dDistSqr > 0)
    {
        double QDist = vDist - sqrt(dDistSqr);

        if(QDist < ray.closestIntersectionDistance && QDist > (1 * pow(10, -5)))
        {
            ray.closestIntersectionDistance = QDist;

            ray.hit = this;
            ray.material = material;
            ray.closestIntersectionPoint = ray.point + ray.direction * QDist;
            ray.surfaceNormal = (ray.closestIntersectionPoint - position).normalize();

            return true;
        }
    }

    return false;
}

__device__ Ray Sphere::makeExitRefrationRay(const Ray &invRay, double originalIndex, double newIndex) const
{
    Vector<3> refractionDirection = doSnellsLaw(invRay.direction, invRay.surfaceNormal, originalIndex, newIndex);
    Vector<3> exitPoint = invRay.closestIntersectionPoint + refractionDirection * 2 * (position - invRay.closestIntersectionPoint).dot(refractionDirection);
    return Ray(exitPoint, doSnellsLaw(-refractionDirection, (position - exitPoint).normalize(), newIndex, originalIndex));
}