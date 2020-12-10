//
// Created by Max Rosoff on 1/29/20.
//

#include "Vector.h"

using namespace std;

Vector::Vector(int size)
{
    data.resize(size);
}

Vector::Vector(const initializer_list<double> &il)
{
    *this = il;
}

Vector &Vector::operator=(const initializer_list<double> &il)
{
    data.resize(il.size());
    copy(il.begin(), il.end(), data.begin());
    return *this;
}

std::ostream &operator<<(std::ostream &out, const Vector &other)
{
    out << "[";

    size_t i = 0;

    for(i = 0; i < other.size() - 1; i++)
    {
        out << other[i] << ", ";
    }

    out << other[i] << "]" << '\n';

    return out;
}
