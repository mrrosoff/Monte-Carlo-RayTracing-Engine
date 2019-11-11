//
// Created by Max Rosoff on 9/7/2019.
//

#include "Remap.h"

using namespace std;

Remap::Remap(const vector<string> &transformData) :

objPath(transformData[9])

{
    Eigen::Vector3d rotationVector;

    rotationVector << stod(transformData[1]), stod(transformData[2]), stod(transformData[3]);

    Eigen::Matrix4d rotation = findRotationMatrix(rotationVector, stod(transformData[4]));

    Eigen::Matrix4d scalar;
    scalar << stod(transformData[5]), 0,                      0,                          0,
              0,                      stod(transformData[5]), 0,                          0,
              0,                      0,                          stod(transformData[5]), 0,
              0,                      0,                          0,                          1;

    Eigen::Matrix4d translation;
    translation << 1, 0, 0, stod(transformData[6]),
                   0, 1, 0, stod(transformData[7]),
                   0, 0, 1, stod(transformData[8]),
                   0, 0, 0, 1;


    transformation = translation * scalar * rotation;
}

Eigen::Matrix4d Remap::findRotationMatrix(Eigen::Vector3d &rotationVector, double theta)
{
    auto cords = changeCords(rotationVector);
    Eigen::Matrix4d zMatrix;

    // Radians!!!

    theta = M_PI * theta / 180;

    zMatrix << cos(theta), -sin(theta), 0, 0,
               sin(theta), cos(theta),  0, 0,
               0,          0,           1, 0,
               0,          0,           0, 1;

    return cords.transpose() * zMatrix * cords;
}


Eigen::Matrix4d Remap::changeCords(Eigen::Vector3d &rotationVector)
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

int Remap::findMinIndex(Eigen::Vector3d &rotationVector)
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