//
// Created by Max Rosoff on 9/21/2019.
//

#include "Sphere.h"

using namespace std;

Sphere::Sphere(const Eigen::Vector3d &position, const double radius, const Material &material) :

position(position), radius(radius)

{
    this->material = material;
}

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
            ray.surfaceNormal = (ray.closestIntersectionPoint - position).normalized();
            ray.material = material;
            return true;
        }
    }

    return false;
}

ostream &operator<<(ostream &out, const Sphere &sph)
{
    Eigen::IOFormat ArrayFormat(Eigen::StreamPrecision, 0, "", ", ", "", "", "[", "]");
    out << "Sphere Position: " << sph.position.format(ArrayFormat) << '\n';
    out << "Sphere Radius: " << sph.radius << '\n';
    out << "Sphere Material: \n" << sph.material << '\n';

    return out;
}
