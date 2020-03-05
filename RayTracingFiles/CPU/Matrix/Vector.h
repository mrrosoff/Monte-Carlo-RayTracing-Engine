//
// Created by Max Rosoff on 1/29/20.
//

#ifndef RAYTRACER_VECTOR_H
#define RAYTRACER_VECTOR_H

#include <iostream>

#include <algorithm>
#include <cmath>
#include <vector>
#include <initializer_list>


class Vector {

public:

    Vector() = default;
    Vector(const Vector &) = default;
    Vector &operator=(const Vector &) = default;
    ~Vector() = default;

    explicit Vector(int);
    Vector(const std::initializer_list<double> &);
    Vector &operator=(const std::initializer_list<double> &);

    inline size_t size() const
    {
        return data.size();
    }

    inline double operator[](const int i) const
    {
        return data[i];
    }

    inline double &operator[](const int i)
    {
        return data[i];
    }

    inline Vector operator-() const
    {
        Vector newVector(data.size());

        for(size_t i = 0; i < data.size(); i++)
        {
            newVector[i] = -data[i];
        }

        return newVector;
    }

    inline Vector operator+(const Vector &other) const
    {
        Vector newVector(data.size());

        for(size_t i = 0; i < data.size(); i++)
        {
            newVector[i] = data[i] + other[i];
        }

        return newVector;
    }

    inline Vector operator-(const Vector &other) const
    {
        Vector newVector(data.size());

        for(size_t i = 0; i < data.size(); i++)
        {
            newVector[i] = data[i] - other[i];
        }

        return newVector;
    }

    inline Vector operator*(const Vector &other) const
    {
        Vector newVector(data.size());

        for(size_t i = 0; i < data.size(); i++)
        {
            newVector[i] = data[i] * other[i];
        }

        return newVector;
    }

    inline Vector &operator+=(const Vector &other)
    {
        *this = operator+(other);
        return *this;
    }

    inline Vector &operator-=(const Vector &other)
    {
        *this = operator-(other);
        return *this;
    }

    inline Vector &operator*=(const Vector &other)
    {
        *this = operator*(other);
        return *this;
    }

    inline Vector operator*(const double scalar) const
    {
        Vector newVector(data.size());

        for(size_t i = 0; i < data.size(); i++)
        {
            newVector[i] = data[i] * scalar;
        }

        return newVector;
    }

    inline Vector operator/(const double scalar) const
    {
        return operator*(1 / scalar);
    }

    inline Vector &operator*=(const double scalar)
    {
        *this = operator*(scalar);
        return *this;
    }

    inline Vector &operator/=(const double scalar)
    {
        *this = operator/(1 / scalar);
        return *this;
    }

    inline double length() const
    {
        return sqrt(this->dot(*this));
    }

    inline Vector normalize() const
    {
        return *this / length();
    }

    inline double dot(const Vector &other) const
    {
        double sum = 0;

        for(size_t i = 0; i < data.size(); i++)
        {
            sum += data[i] * other[i];
        }

        return sum;
    }

    inline Vector cross(const Vector &other) const
    {
        return {data[1] * other[2] - data[2] * other[1],
              -(data[0] * other[2] - data[2] * other[0]),
                data[0] * other[1] - data[1] * other[0]};
    }

private:

    std::vector<double> data;
};

inline Vector operator*(double scalar, const Vector &other)
{
    return other * scalar;
}

std::ostream &operator<<(std::ostream &, const Vector &);



#endif //RAYTRACER_VECTOR_H
