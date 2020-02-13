//
// Created by Max Rosoff on 1/29/20.
//

#include "Matrix.h"

using namespace std;

Matrix::Matrix(int rows, int cols)
{
    data.reserve(rows);

    for(int i = 0; i < rows; i++)
    {
        data[i] = Vector(cols);
    }
}

Matrix::Matrix(const Vector &one, const Vector &two, const Vector &three)
{
    data.reserve(3);

    data[0] = one;
    data[1] = two;
    data[2] = three;
}

Matrix::Matrix(const std::initializer_list<Vector> &il)
{
    *this = il;
}

Matrix &Matrix::operator=(const std::initializer_list<Vector> &il)
{
    data.reserve(il.size());

    auto it = il.begin();

    for(int i = 0; i < il.size(); i++)
    {
        data[i] = *it;
        it++;
    }

    return *this;
}

std::ostream &operator<<(std::ostream &out, const Matrix &other)
{
    out << other;
    return out;
}
