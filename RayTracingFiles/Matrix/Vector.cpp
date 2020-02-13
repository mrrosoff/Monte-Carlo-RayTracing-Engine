//
// Created by Max Rosoff on 1/29/20.
//

#include "Vector.h"

using namespace std;

Vector::Vector()
{
    data.reserve(3);
}

Vector::Vector(double size)
{
    data.reserve(size);
}

Vector::Vector(double x, double y, double z)
{
    data.reserve(3);

    data[0] = x;
    data[1] = y;
    data[2] = z;
}

Vector::Vector(const initializer_list<double> &il)
{
    *this = il;
}

Vector &Vector::operator=(const initializer_list<double> &il)
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

std::ostream &operator<<(std::ostream &out, const Vector &vector)
{
    out << "[" << vector[0] << ", " << vector[1] << ", " << vector[2] << "]" << '\n';

    return out;
}
