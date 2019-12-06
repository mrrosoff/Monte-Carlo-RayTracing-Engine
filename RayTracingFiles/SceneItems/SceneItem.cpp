//
// Created by Max Rosoff on 11/18/2019.
//

#include "SceneItem.h"

using namespace std;

Eigen::Vector3d SceneItem::doSnellsLaw(const Eigen::Vector3d &direction, const Eigen::Vector3d &surfaceNormal, const double indexOne, const double indexTwo) const
{
    double refractionEquation  = indexOne / indexTwo;

    double WN = direction.dot(surfaceNormal);
    double indexSq = pow(refractionEquation, 2) * (pow(WN, 2) - 1) + 1;

    if (indexSq < 0)
    {
        throw range_error("Refraction Not Needed.");
    }

    else
    {
        double beta = (refractionEquation * WN) - sqrt(indexSq);
        return -refractionEquation * direction + beta * surfaceNormal;
    }
}
