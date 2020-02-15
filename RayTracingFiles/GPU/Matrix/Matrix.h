//
// Created by Max Rosoff on 1/29/20.
//

#ifndef RAYTRACER_MATRIX_H
#define RAYTRACER_MATRIX_H

#include <cmath>
#include <iostream>
#include <vector>
#include <initializer_list>

#include "./Vector.h"

class Matrix {

public:

    Matrix() = default;
    Matrix(const Matrix &) = default;
    Matrix &operator=(const Matrix &) = default;
    ~Matrix() = default;

    __host__ __device__ Matrix(int, int);
    __host__ __device__ Matrix(const std::initializer_list<Vector> &);
    __host__ __device__ Matrix &operator=(const std::initializer_list<Vector> &);

    __host__ __device__ inline size_t size() const
    {
        return data.size();
    }

    __host__ __device__ inline Vector operator[](const int i) const
    {
        return data[i];
    }

    __host__ __device__ inline Vector &operator[](const int i)
    {
        return data[i];
    }

    __host__ __device__ inline Matrix operator+(const Matrix &other) const
    {
        Matrix newMatrix(data.size(), data.size());

        for(size_t i = 0; i < data.size(); i++)
        {
            newMatrix[i] = data[i] + other[i];
        }

        return newMatrix;
    }

    __host__ __device__ inline Matrix operator-(const Matrix &other) const
    {
        Matrix newMatrix(data.size(), data.size());

        for(size_t i = 0; i < data.size(); i++)
        {
            newMatrix[i] = data[i] - other[i];
        }

        return newMatrix;
    }

    __host__ __device__ inline Matrix operator*(const Matrix &other) const
    {
        Matrix newMatrix(data.size(), data.size());

        for(size_t i = 0; i < data.size(); i++)
        {
            for(size_t j = 0; j < data.size(); j++)
            {
                for(size_t k = 0; k < data.size(); k++)
                {
                    newMatrix[i][j] += data[i][k] * other[k][j];
                }
            }
        }

        return newMatrix;
    }

    __host__ __device__ inline Matrix &operator+=(const Matrix &other)
    {
        *this = operator+(other);
        return *this;
    }

    __host__ __device__ inline Matrix &operator-=(const Matrix &other)
    {
        *this = operator-(other);
        return *this;
    }

    __host__ __device__ inline Matrix &operator*=(const Matrix &other)
    {
        *this = operator*(other);
        return *this;
    }

    __host__ __device__ inline Vector operator*(const Vector &other) const
    {
        Vector newVector(other.size());

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
        Matrix newMatrix(data.size(), data.size());

        for(size_t i = 0; i < data.size(); i++)
        {
            for (size_t j = 0; j < data.size(); j++)
            {
                newMatrix[j][i] = data[i][j];
            }
        }

        return newMatrix;
    }

    __host__ __device__ inline double determinant()
    {
        double det = 0;

        if(data.size() != 3)
        {
            throw std::invalid_argument("Only 3x3 Matrices Supported For Determinant!");
        }

        for(int i = 0; i < 3; i++)
        {
            det += data[0][i] * (data[1][(i + 1) % 3] * data[2][(i + 2) % 3] - data[1][(i + 2) % 3] * data[2][(i + 1) % 3]);
        }

        return det;
    }

    __host__ __device__ inline Matrix inverse()
    {
        Matrix newMatrix(data.size(), data.size());

        if(data.size() != 3)
        {
            throw std::invalid_argument("Only 3x3 Matrices Supported For Inverse!");
        }

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

    std::vector<Vector> data;
};

__host__ std::ostream &operator<<(std::ostream &, const Matrix &);


#endif //RAYTRACER_MATRIX_H
