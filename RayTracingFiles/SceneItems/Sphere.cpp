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
            ray.closestIntersectionPoint = ray.point + QDist * ray.direction;
            ray.surfaceNormal = (ray.closestIntersectionPoint - position).normalized();

            return true;
        }
    }

    return false;
}

Ray Sphere::makeExitRefrationRay(const Ray &invRay, double originalIndex, double newIndex) const
{
    Eigen::Vector3d refractionDirection = doSnellsLaw(invRay.direction, invRay.surfaceNormal, originalIndex, newIndex);
    Eigen::Vector3d exitPoint = invRay.closestIntersectionPoint + 2 * (position - invRay.closestIntersectionPoint).dot(refractionDirection) * refractionDirection;
    return Ray(exitPoint, doSnellsLaw(-1 * refractionDirection, (position - exitPoint).normalized(), newIndex, originalIndex));
}

ostream &operator<<(ostream &out, const Sphere &sph)
{
    Eigen::IOFormat ArrayFormat(Eigen::StreamPrecision, 0, "", ", ", "", "", "[", "]");

    out << "Sphere Position: " << sph.position.format(ArrayFormat) << '\n';
    out << "Sphere Radius: " << sph.radius << '\n';
    out << "Sphere Material: \n" << sph.material << '\n';

    return out;
}
