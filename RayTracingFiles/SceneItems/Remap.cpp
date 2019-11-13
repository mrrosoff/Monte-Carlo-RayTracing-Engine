//
// Created by Max Rosoff on 9/7/2019.
//

#include "Remap.h"

using namespace std;

Remap::Remap(const Eigen::Vector3d &rotationVector, const double theta, const Eigen::Matrix4d &scalar, const Eigen::Matrix4d &translation, const double smoothingAngle, const string &path) :

objPath(path), smoothingAngle(smoothingAngle)

{
    Eigen::Matrix4d rotation = findRotationMatrix(rotationVector, theta);
    transformation = translation * scalar * rotation;
}

Eigen::Matrix4d Remap::findRotationMatrix(const Eigen::Vector3d &rotationVector, const double theta) const
{
    auto coords = changeCords(rotationVector);
    Eigen::Matrix4d zMatrix;

    double radTheta = M_PI * theta / 180;

    zMatrix << cos(radTheta), -sin(radTheta), 0, 0,
               sin(radTheta), cos(radTheta),  0, 0,
               0,             0,              1, 0,
               0,             0,              0, 1;

    return coords.transpose() * zMatrix * coords;
}

Eigen::Matrix4d Remap::changeCords(const Eigen::Vector3d &rotationVector) const
{
    auto bottomVector = rotationVector.normalized();

    auto helperVector = bottomVector;
    int min = findMinIndex(bottomVector);
    helperVector[min] = 1;

    auto topVector = bottomVector.cross(helperVector);
    topVector.normalize();

    auto middleVector = bottomVector.cross(topVector);

    Eigen::Matrix4d r;

    r << topVector[0],    topVector[1],    topVector[2],    0,
         middleVector[0], middleVector[1], middleVector[2], 0,
         bottomVector[0], bottomVector[1], bottomVector[2], 0,
         0,               0,               0,               1;

    return r;
}

int Remap::findMinIndex(const Eigen::Vector3d &rotationVector) const
{
    int index = 0;
    double min = numeric_limits<double>::max();

    for(int i = 0; i < 3; i++)
    {
        double currentElement = rotationVector[i];

        if(currentElement < min)
        {
            min = currentElement;
            index = i;
        }
    }

    return index;
}

ostream &operator<<(ostream &out, const Remap &remap)
{
    out << "Remap Path: " << remap.objPath << '\n';
    out << "Remap Transformation Matrix:\n" << remap.transformation << '\n';

    return out;
}
