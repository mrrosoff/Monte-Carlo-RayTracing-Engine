//
// Created by Max Rosoff on 1/29/20.
//

#include "Matrix.h"

using namespace std;

Matrix::Matrix(int rows, int cols)
{
    data.resize(rows);

    for(int i = 0; i < rows; i++)
    {
        data[i] = Vector(cols);
    }
}

Matrix::Matrix(const std::initializer_list<Vector> &il)
{
    *this = il;
}

Matrix &Matrix::operator=(const std::initializer_list<Vector> &il)
{
    data.resize(il.size());
    copy(il.begin(), il.end(), data.begin());
    return *this;
}

std::ostream &operator<<(std::ostream &out, const Matrix &other)
{
    for(size_t i = 0; i < other.size(); i++)
    {
        out << other[i];
    }

    return out;
}
