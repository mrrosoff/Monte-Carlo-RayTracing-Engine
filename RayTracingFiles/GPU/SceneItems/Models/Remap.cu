//
// Created by Max Rosoff on 9/7/2019.
//

#include "Remap.h"

using namespace std;

Remap::Remap(const Vector<3> &rotationVector, const double theta, const Matrix<4, 4> &scalar, const Matrix<4, 4> &translation, const double smoothingAngle, string &path) :

smoothingAngle(smoothingAngle), path(path)

{
    Matrix<4, 4> rotation = findRotationMatrix(rotationVector, theta);
    transformation = translation * scalar * rotation;
}

Matrix<4, 4> Remap::findRotationMatrix(const Vector<3> &rotationVector, const double theta) const
{
    auto coords = changeCords(rotationVector);
    double radTheta = M_PI * theta / 180;

    Matrix<4, 4> zMatrix;

    Vector<4> zMatrixLineOne;
    Vector<4> zMatrixLineTwo;
    Vector<4> zMatrixLineThree;
    Vector<4> zMatrixLineFour;

    zMatrixLineOne[0] = cos(radTheta);
    zMatrixLineOne[1] = -sin(radTheta);
    zMatrixLineOne[2] = 0;
    zMatrixLineOne[3] = 0;

    zMatrixLineTwo[0] = sin(radTheta);
    zMatrixLineTwo[1] = cos(radTheta);
    zMatrixLineTwo[2] = 0;
    zMatrixLineTwo[3] = 0;

    zMatrixLineThree[0] = 0;
    zMatrixLineThree[1] = 0;
    zMatrixLineThree[2] = 1;
    zMatrixLineThree[3] = 0;

    zMatrixLineFour[0] = 0;
    zMatrixLineFour[1] = 0;
    zMatrixLineFour[2] = 0;
    zMatrixLineFour[3] = 1;

    zMatrix[0] = zMatrixLineOne;
    zMatrix[1] = zMatrixLineTwo;
    zMatrix[2] = zMatrixLineThree;
    zMatrix[3] = zMatrixLineFour;

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

    Vector<4> rLineOne;
    Vector<4> rLineTwo;
    Vector<4> rLineThree;
    Vector<4> rLineFour;

    rLineOne[0] = topVector[0];
    rLineOne[1] = topVector[1];
    rLineOne[2] = topVector[2];
    rLineOne[3] = 0;

    rLineTwo[0] = middleVector[0];
    rLineTwo[1] = middleVector[1];
    rLineTwo[2] = middleVector[2];
    rLineTwo[3] = 0;

    rLineThree[0] = bottomVector[0];
    rLineThree[1] = bottomVector[1];
    rLineThree[2] = bottomVector[2];
    rLineThree[3] = 0;

    rLineFour[0] = 0;
    rLineFour[1] = 0;
    rLineFour[2] = 0;
    rLineFour[3] = 1;

    r[0] = rLineOne;
    r[1] = rLineTwo;
    r[2] = rLineThree;
    r[3] = rLineFour;

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
