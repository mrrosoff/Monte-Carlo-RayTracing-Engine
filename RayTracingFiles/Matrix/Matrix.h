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

    Matrix(int, int);
    Matrix(const Vector &, const Vector &, const Vector &);

    Matrix(const std::initializer_list<Vector> &);
    Matrix &operator=(const std::initializer_list<Vector> &);

    inline Vector operator[](const int i) const
    {
        return data[i];
    }

    inline Vector &operator[](const int i)
    {
        return data[i];
    }

    inline Matrix operator+(const Matrix &other) const
    {
        Matrix newMatrix(data.size(), data[0].size());

        for(size_t i = 0; i < data.size(); i++)
        {
            newMatrix[i] = data[i] + other[i];
        }

        return newMatrix;
    }

    inline Matrix operator-(const Matrix &other) const
    {
        Matrix newMatrix(data.size(), data[0].size());

        for(size_t i = 0; i < data.size(); i++)
        {
            newMatrix[i] = data[i] - other[i];
        }

        return newMatrix;
    }

    inline Matrix operator*(const Matrix &other) const
    {
        Matrix newMatrix(data.size(), data[0].size());

        for(size_t i = 0; i < data.size(); i++)
        {
            for(size_t j = 0; j < other[0].size(); j++)
            {
                for(size_t k = 0; k < data[0].size(); k++)
                {
                    newMatrix[i][j] += data[i][k] * other[k][j];
                }
            }
        }

        return newMatrix;
    }

    inline Matrix &operator+=(const Matrix &other)
    {
        for(size_t i = 0; i < data.size(); i++)
        {
            data[i] += other[i];
        }

        return *this;
    }

    inline Matrix &operator-=(const Matrix &other)
    {
        for(size_t i = 0; i < data.size(); i++)
        {
            data[i] -= other[i];
        }

        return *this;
    }

    inline Matrix &operator*=(const Matrix &other)
    {
        Matrix newMatrix = *this * other;

        for(size_t i = 0; i < data.size(); i++)
        {
            data[i] = newMatrix[i];
        }

        return *this;
    }

    inline Vector operator*(const Vector &other) const
    {
        Vector newVector(other.size());

        for (size_t i = 0;i < newVector.size(); i++)
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

    inline Vector operator/(const Vector &other) const
    {
        Vector newVector(other.size());

        for (size_t i = 0;i < newVector.size(); i++)
        {
            newVector[i] = 0;
        }

        for(size_t i = 0; i < newVector.size(); i++)
        {
            for(size_t j = 0; j < newVector.size(); j++)
            {
                newVector[i] += data[i][j] / other[j];
            }
        }

        return newVector;
    }

    inline Matrix transpose()
    {
        Matrix newMatrix(data.size(), data[0].size());

        for(size_t i = 0; i < data.size(); i++)
        {
            for (size_t j = 0; j < data[0].size(); j++)
            {
                newMatrix[j][i] = data[i][j];
            }
        }

        return newMatrix;
    }

    inline Matrix inverse()
    {
        Matrix newMatrix(data.size(), data[0].size());

        double det = determinant(newMatrix, data.size());

        if (det == 0)
        {
            throw std::invalid_argument("Not Invertible");
        }

        Matrix adj = adjoint(*this);

        for (size_t i = 0; i < data.size(); i++)
        {
            for (size_t j = 0; j < data.size(); j++)
            {
                newMatrix[i][j] = adj[i][j] / det;
            }
        }

        return newMatrix;
    }

private:

    std::vector<Vector> data;

    static inline void getCofactor(const Matrix &A, Matrix &temp, const int p, const int q, const int n)
    {
        int i = 0, j = 0;

        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < n; col++)
            {
                if (row != p && col != q)
                {
                    temp[i][j++] = A[row][col];

                    if (j == n - 1)
                    {
                        j = 0;
                        i++;
                    }
                }
            }
        }
    }

    static inline double determinant(const Matrix &A, const size_t n)
    {
        double determinantValue = 0;

        if (n == 1)
        {
            return A[0][0];
        }

        int sign = 1;
        Matrix temp(A[0].size(), A[0].size());

        for (size_t f = 0; f < n; f++)
        {
            getCofactor(A, temp, 0, f, n);
            determinantValue += sign * A[0][f] * determinant(temp, n - 1);
            sign = -sign;
        }

        return determinantValue;
    }

    static inline Matrix adjoint(const Matrix &A)
    {
        Matrix newMatrix(A[0].size(), A[0].size());

        if (A[0].size() == 1)
        {
            newMatrix[0][0] = 1;
            return newMatrix;
        }

        int sign;
        Matrix temp(A[0].size(), A[0].size());

        for (size_t i = 0; i < A[0].size(); i++)
        {
            for (size_t j=0; j < A[0].size(); j++)
            {
                getCofactor(A, temp, i, j, A[0].size());

                if ((i + j) % 2 == 0)
                {
                    sign = 1;
                }

                else
                {
                    sign = -1;
                }

                newMatrix[j][i] = sign * determinant(temp, A[0].size() - 1);
            }
        }

        return newMatrix;
    }
};

std::ostream &operator<<(std::ostream &, const Matrix &);


#endif //RAYTRACER_MATRIX_H
