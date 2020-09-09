//
// Created by Max Rosoff on 1/29/20.
//

#ifndef RAYTRACER_MATRIX_H
#define RAYTRACER_MATRIX_H

#include <cmath>

#include "./Vector.h"

template <int w, int h>
class Matrix {

public:

    Matrix() = default;
    Matrix(const Matrix &) = default;
    Matrix &operator=(const Matrix &) = default;
    ~Matrix() = default;

    __host__ __device__ inline size_t size() const
    {
        return h;
    }

    __host__ __device__ inline Vector<w> operator[](const int i) const
    {
        return data[i];
    }

    __host__ __device__ inline Vector<w> &operator[](const int i)
    {
        return data[i];
    }

    __host__ __device__ inline Matrix<w, h> operator+(const Matrix<w, h> &other) const
    {
        Matrix<w, h> newMatrix;

        for(size_t i = 0; i < size(); i++)
        {
            newMatrix[i] = data[i] + other[i];
        }

        return newMatrix;
    }

    __host__ __device__ inline Matrix<w, h> operator-(const Matrix<w, h> &other) const
    {
        Matrix<w, h> newMatrix;

        for(size_t i = 0; i < size(); i++)
        {
            newMatrix[i] = data[i] - other[i];
        }

        return newMatrix;
    }

    __host__ __device__ inline Matrix<w, h> operator*(const Matrix<w, h> &other) const
    {
        Matrix<w, h> newMatrix;

        for(size_t i = 0; i < size(); i++)
        {
            for(size_t j = 0; j < size(); j++)
            {
                for(size_t k = 0; k < size(); k++)
                {
                    newMatrix[i][j] += data[i][k] * other[k][j];
                }
            }
        }

        return newMatrix;
    }

    __host__ __device__ inline Matrix<w, h> &operator+=(const Matrix<w, h> &other)
    {
        *this = operator+(other);
        return *this;
    }

    __host__ __device__ inline Matrix<w, h> &operator-=(const Matrix<w, h> &other)
    {
        *this = operator-(other);
        return *this;
    }

    __host__ __device__ inline Matrix<w, h> &operator*=(const Matrix<w, h> &other)
    {
        *this = operator*(other);
        return *this;
    }

    __host__ __device__ inline Vector<w> operator*(const Vector<w> &other) const
    {
        Vector<w> newVector;

        for (size_t i = 0; i < newVector.size(); i++)
        {
            newVector[i] = 0;
        }

        for(size_t i = 0; i < newVector.size(); i++)
        {
            for(size_t j = 0; j < newVector.size(); j++)
            {
                newVector[i] += data[i][j] * other[j];
            }
        }

        return newVector;
    }

    __host__ __device__ inline Matrix transpose()
    {
        Matrix<w, h> newMatrix;

        for(size_t i = 0; i < size(); i++)
        {
            for (size_t j = 0; j < size(); j++)
            {
                newMatrix[j][i] = data[i][j];
            }
        }

        return newMatrix;
    }

    __host__ __device__ inline double determinant()
    {
        double det = 0;

        for(int i = 0; i < 3; i++)
        {
            det += data[0][i] * (data[1][(i + 1) % 3] * data[2][(i + 2) % 3] - data[1][(i + 2) % 3] * data[2][(i + 1) % 3]);
        }

        return det;
    }

    __host__ __device__ inline Matrix inverse()
    {
        Matrix<3, 3> newMatrix;

        double det = determinant();

        if (det == 0)
        {
            throw std::invalid_argument("Not Invertible");
        }

        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                newMatrix[i][j] = (data[(j + 1) % 3][(i + 1) % 3] * data[(j + 2) % 3][(i + 2) % 3] -
                                   data[(j + 1) % 3][(i + 2) % 3] * data[(j + 2) % 3][(i + 1) % 3]) / det;
            }
        }

        return newMatrix;
    }

private:

    Vector<w> data[h];
};


#endif //RAYTRACER_MATRIX_H
