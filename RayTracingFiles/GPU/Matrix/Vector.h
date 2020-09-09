//
// Created by Max Rosoff on 1/29/20.
//

#ifndef RAYTRACER_VECTOR_H
#define RAYTRACER_VECTOR_H

#include <cmath>

template <int n>
class Vector {

public:

    Vector() = default;
    Vector(const Vector &) = default;
    Vector &operator=(const Vector &) = default;
    ~Vector() = default;

    __host__ __device__ inline size_t size() const
    {
        return n;
    }

    __host__ __device__ inline double operator[](const int i) const
    {
        return data[i];
    }

    __host__ __device__ inline double &operator[](const int i)
    {
        return data[i];
    }

    __host__ __device__ inline Vector<n> operator-() const
    {
        Vector<n> newVector;

        for(size_t i = 0; i < size(); i++)
        {
            newVector[i] = -data[i];
        }

        return newVector;
    }

    __host__ __device__ inline Vector<n> operator+(const Vector<n> &other) const
    {
        Vector<n> newVector;

        for(size_t i = 0; i < size(); i++)
        {
            newVector[i] = data[i] + other[i];
        }

        return newVector;
    }

    __host__ __device__ inline Vector<n> operator-(const Vector<n> &other) const
    {
        Vector<n> newVector;

        for(size_t i = 0; i < size(); i++)
        {
            newVector[i] = data[i] - other[i];
        }

        return newVector;
    }

    __host__ __device__ inline Vector<n> operator*(const Vector<n> &other) const
    {
        Vector<n> newVector;

        for(size_t i = 0; i < size(); i++)
        {
            newVector[i] = data[i] * other[i];
        }

        return newVector;
    }

    __host__ __device__ inline Vector<n> &operator+=(const Vector<n> &other)
    {
        *this = operator+(other);
        return *this;
    }

    __host__ __device__ inline Vector<n> &operator-=(const Vector<n> &other)
    {
        *this = operator-(other);
        return *this;
    }

    __host__ __device__ inline Vector<n> &operator*=(const Vector<n> &other)
    {
        *this = operator*(other);
        return *this;
    }

    __host__ __device__ inline Vector<n> operator*(const double scalar) const
    {
        Vector<n> newVector;

        for(size_t i = 0; i < size(); i++)
        {
            newVector[i] = data[i] * scalar;
        }

        return newVector;
    }

    __host__ __device__ inline Vector<n> operator/(const double scalar) const
    {
        return operator*(1 / scalar);
    }

    __host__ __device__ inline Vector<n> &operator*=(const double scalar)
    {
        *this = operator*(scalar);
        return *this;
    }

    __host__ __device__ inline Vector<n> &operator/=(const double scalar)
    {
        *this = operator/(1 / scalar);
        return *this;
    }

    __host__ __device__ inline double length() const
    {
        return sqrt(this->dot(*this));
    }

    __host__ __device__ inline Vector<n> normalize() const
    {
        return *this / length();
    }

    __host__ __device__ inline double dot(const Vector<n> &other) const
    {
        double sum = 0;

        for(size_t i = 0; i < size(); i++)
        {
            sum += data[i] * other[i];
        }

        return sum;
    }

    __host__ __device__ inline Vector<3> cross(const Vector<3> &other) const
    {
        Vector<3> newVector;

        newVector[0] = data[1] * other[2] - data[2] * other[1];
        newVector[1] = -(data[0] * other[2] - data[2] * other[0]);
        newVector[2] = data[0] * other[1] - data[1] * other[0];

        return newVector;
    }

private:

    double data[n];
};


#endif //RAYTRACER_VECTOR_H
