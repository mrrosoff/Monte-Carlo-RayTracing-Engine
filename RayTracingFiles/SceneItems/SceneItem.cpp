//
// Created by Max Rosoff on 11/18/2019.
//

#include "SceneItem.h"

using namespace std;

Vector SceneItem::doSnellsLaw(const Vector &direction, const Vector &surfaceNormal, const double indexOne, const double indexTwo) const
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
        return direction * -refractionEquation + surfaceNormal * beta;
    }
}
