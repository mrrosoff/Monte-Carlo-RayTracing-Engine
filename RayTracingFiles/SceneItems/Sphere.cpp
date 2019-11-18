//
// Created by Max Rosoff on 9/21/2019.
//

#include "Sphere.h"

using namespace std;

Sphere::Sphere(const Eigen::Vector3d &position, const double radius, const Material &material) :

position(position), radius(radius), material(material)

{}

bool Sphere::intersectionTest(Ray &ray) const
{
    auto cVector = position - ray.point;
    auto cDist = cVector.dot(cVector);
    auto vDist = cVector.dot(ray.direction);
    double dDistSqr = radius * radius - (cDist - vDist * vDist);

    const double EPSILON = 1 * pow(10, -5);
    
    if(dDistSqr < 0)
    {
        return false;
    }

    else
    {
        double QDist = vDist - sqrt(dDistSqr);

        if(QDist < ray.closestIntersectionDistance && QDist > EPSILON)
        {
            ray.closestIntersectionDistance = QDist;
            ray.closestIntersectionPoint = ray.point + QDist * ray.direction;

            ray.hit = this;
            ray.surfaceNormal = (ray.closestIntersectionPoint - position).normalized();
            ray.material = material;

            return true;
        }
    }

    return false;
}

Ray Sphere::makeExitRefrationRay(const Ray &invRay, double indexOne, double indexTwo) const
{
    Eigen::Vector3d refractPoint = doSnellsLaw(invRay, (invRay.point - position).normalized(), indexTwo, indexOne);
    Eigen::Vector3d exitPoint = invRay.point + 2 * (position - invRay.point).dot(refractPoint) * refractPoint;
    Ray refractRay(exitPoint, -refractPoint);
    return Ray(exitPoint, doSnellsLaw(refractRay, (position - (invRay.point + 2 * (position - invRay.point).dot(refractPoint) * refractPoint)).normalized(), indexOne, indexTwo));
}

Eigen::Vector3d Sphere::doSnellsLaw(const Ray &invRay, const Eigen::Vector3d &normal, const double indexOne, const double indexTwo) const
{
    double refractionEquation  = indexOne / indexTwo;

    double wn = invRay.direction.dot(normal);
    double indexSq = refractionEquation * refractionEquation * (wn * wn - 1) + 1;

    if (indexSq < 0)
    {
        throw range_error("Refraction Error");
    }

    else
    {
        double b = (refractionEquation * wn) - sqrt(indexSq);
        return -refractionEquation * invRay.direction + b * normal;
    }
}

ostream &operator<<(ostream &out, const Sphere &sph)
{
    Eigen::IOFormat ArrayFormat(Eigen::StreamPrecision, 0, "", ", ", "", "", "[", "]");

    out << "Sphere Position: " << sph.position.format(ArrayFormat) << '\n';
    out << "Sphere Radius: " << sph.radius << '\n';
    out << "Sphere Material: \n" << sph.material << '\n';

    return out;
}
