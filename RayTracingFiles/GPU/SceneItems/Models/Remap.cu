//
// Created by Max Rosoff on 9/7/2019.
//

#include "Remap.h"

using namespace std;

Remap::Remap(const Vector<3> &rotationVector, const double theta, const Matrix<4, 4> &scalar, const Matrix<4, 4> &translation, const double smoothingAngle, const string &path) :

smoothingAngle(smoothingAngle), objPath(path), transformation(Matrix(4, 4))

{
    Matrix<4, 4> rotation = findRotationMatrix(rotationVector, theta);
    transformation = translation * scalar * rotation;
}

Matrix<4, 4> Remap::findRotationMatrix(const Vector<3> &rotationVector, const double theta) const
{
    auto coords = changeCords(rotationVector);
    Matrix<4, 4> zMatrix;

    double radTheta = M_PI * theta / 180;

    zMatrix[0] = {cos(radTheta), -sin(radTheta), 0, 0};
    zMatrix[1] = {sin(radTheta), cos(radTheta),  0, 0};
    zMatrix[2] = {0, 0, 1, 0};
    zMatrix[3] = {0, 0, 0, 1};

    return coords.transpose() * zMatrix * coords;
}

Matrix<4, 4> Remap::changeCords(const Vector<3> &rotationVector) const
{
    auto bottomVector = rotationVector.normalize();

    auto helperVector = bottomVector;
    int min = findMinIndex(bottomVector);
    helperVector[min] = 1;

    auto topVector = bottomVector.cross(helperVector);
    topVector.normalize();

    auto middleVector = bottomVector.cross(topVector);

    Matrix<4, 4> r;

    r[0] = {topVector[0], topVector[1], topVector[2], 0};
    r[1] = {middleVector[0], middleVector[1], middleVector[2], 0};
    r[2] = {bottomVector[0], bottomVector[1], bottomVector[2], 0};
    r[3] = {0, 0, 0, 1};

    return r;
}

int Remap::findMinIndex(const Vector<3> &rotationVector) const
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
