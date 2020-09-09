//
// Created by Max Rosoff on 11/18/2019.
//

#include "SceneItem.h"

using namespace std;

__device__ Vector<3> SceneItem::doSnellsLaw(const Vector<3> &direction, const Vector<3> &surfaceNormal, const double indexOne, const double indexTwo) const
{
    double refractionEquation  = indexOne / indexTwo;

    double WN = direction.dot(surfaceNormal);
    double indexSq = pow(refractionEquation, 2) * (pow(WN, 2) - 1) + 1;

    if (indexSq < 0)
    {
        Vector<3> badReturnValue;

        badReturnValue[0] = 0;
        badReturnValue[1] = 0;
        badReturnValue[2] = 0;

        return badReturnValue;
    }

    else
    {
        double beta = (refractionEquation * WN) - sqrt(indexSq);
        return direction * -refractionEquation + surfaceNormal * beta;
    }
}
