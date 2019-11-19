//
// Created by Max Rosoff on 11/18/2019.
//

#include "SceneItem.h"

using namespace std;

Eigen::Vector3d SceneItem::doSnellsLaw(const Ray &invRay, const double indexOne, const double indexTwo) const
{
    double refractionEquation  = indexOne / indexTwo;

    double WN = invRay.direction.dot(invRay.surfaceNormal);
    double indexSq = pow(refractionEquation, 2) * (pow(WN, 2) - 1) + 1;

    if (indexSq < 0)
    {
        throw range_error("Refraction Not Needed.");
    }

    else
    {
        double beta = (refractionEquation * WN) - sqrt(indexSq);
        return -refractionEquation * invRay.direction + beta * invRay.surfaceNormal;
    }
}