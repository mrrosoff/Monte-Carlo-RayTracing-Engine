//
// Created by Max Rosoff on 9/21/2019.
//

#include <iostream>
#include <limits>

#include "DReader.h"
#include "PWriter.h"

using namespace std;

double findRayX(const int row, const double left, const double right, const int width)
{
    return static_cast<double>(row) / (width - 1) * (right - left) + left;
}

double findRayY(const int col, const double top, const double bottom, const int height)
{
    return static_cast<double>(col) / (height - 1) * (bottom - top) + top;
}

vector<Eigen::Vector3d> pixelRay(const int row, const int col, const DReader &driver,
                                 const Eigen::Vector3d &normalizedCameraWVector,
                                 const Eigen::Vector3d &normalizedCameraUVector,
                                 const Eigen::Vector3d &cameraVVector)
{
    auto bounds = driver.bounds;

    double left = bounds[0];
    double right = bounds[1];
    double top = bounds[2];
    double bottom = bounds[3];

    auto resolution = driver.resolution;

    int width = resolution[0];
    int height = resolution[1];

    auto xValue = findRayX(row, left, right, width);
    auto yValue = findRayY(col, top, bottom, height);

    auto point = driver.eye + (driver.focalLength * normalizedCameraWVector) + (xValue * normalizedCameraUVector) + (yValue * cameraVVector);
    auto direction = point - driver.eye;
    auto normalizedDirection = direction.normalized();

    return {point, normalizedDirection};
}

double findDDistSqr(const double radius, const double cDist, const double vDist)
{
    return radius * radius - (cDist - vDist * vDist);
}

Eigen::Vector3d sphereHitPoint(const Eigen::Vector3d &point, const Eigen::Vector3d &direction, const double vDist, const double dDistSqr)
{
    return point + (vDist - sqrt(dDistSqr)) * direction;
}

Eigen::Vector3d raySphereRGB(const DReader &driver, const vector<Eigen::Vector3d> &ray, const Sphere &sphere)
{
    double radius  = sphere.radius;
    auto spherePosition = sphere.position;

    auto point = ray[0];
    auto direction = ray[1];

    auto cVector = spherePosition - point;
    auto cDist = cVector.dot(cVector);

    auto vDist = cVector.dot(direction);

    double dDistSqr = findDDistSqr(radius, cDist, vDist);

    if(dDistSqr < 0)
    {
        return {0, 0, 0};
    }

    else
    {
        auto hitPoint = sphereHitPoint(point, direction, vDist, dDistSqr);

        auto surfaceNormal = hitPoint - sphere.position;
        auto normalizedSurfaceNormal = surfaceNormal.normalized();

        auto r = driver.ambientLight[0] * sphere.Ka[0];
        auto g = driver.ambientLight[1] * sphere.Ka[1];
        auto b = driver.ambientLight[2] * sphere.Ka[2];

        Eigen::Vector3d color(r, g, b);

        for(const auto &light : driver.lights)
        {
            auto lightPoint = light.position;
            auto lightColor = light.rgb;

            auto lightVector = lightPoint - hitPoint;
            auto normalizedLightVector = lightVector.normalized();

            if (normalizedSurfaceNormal.dot(normalxxizedLightVector) > 0.0)
            {
                color[0] += sphere.Kd[0] * lightColor[0] * normalizedSurfaceNormal.dot(normalizedLightVector);
                color[1] += sphere.Kd[1] * lightColor[1] * normalizedSurfaceNormal.dot(normalizedLightVector);
                color[2] += sphere.Kd[2] * lightColor[2] * normalizedSurfaceNormal.dot(normalizedLightVector);
            }

        }

        return color;
    }
}

Eigen::Vector3d findNearestSphere(DReader &driver, const vector<Eigen::Vector3d> &ray)
{
    Sphere* closestSphere;
    double minMagnitude = numeric_limits<double>::max();

    for(auto &sphere : driver.spheres)
    {
        double radius  = sphere.radius;
        auto spherePosition = sphere.position;

        auto point = ray[0];
        auto direction = ray[1];

        auto cVector = spherePosition - point;
        auto cDist = cVector.dot(cVector);

        auto vDist = cVector.dot(direction);

        double dDistSqr = findDDistSqr(radius, cDist, vDist);

        if(dDistSqr < 0)
        {
            continue;
        }

        else
        {
            auto hitPoint = sphereHitPoint(point, direction, vDist, dDistSqr);
            auto magnitude = hitPoint.norm();

            if(magnitude < minMagnitude)
            {
                minMagnitude = magnitude;
                closestSphere = &sphere;
            }
        }
    }

    return raySphereRGB(driver,ray, *closestSphere);
}

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        cerr << "Usage: ./raytracer" << " " << "[Driver File]" << " " << "[PPM Output File]" << '\n';
        return 1;
    }

    // Initialize Driver Reader Object. This will load public members of the driver file for use.

    try
    {
        DReader driver;

        driver << argv[1];

        auto resolution = driver.resolution;

        int width = resolution[0];
        int height = resolution[1];

        auto cameraWVector = driver.eye - driver.lookAtPoint;
        auto normalizedCameraWVector = cameraWVector.normalized();

        auto cameraUVector = driver.upVector.cross(normalizedCameraWVector);
        auto normalizedCameraUVector = cameraUVector.normalized();

        auto cameraVVector = normalizedCameraWVector.cross(normalizedCameraUVector);

        vector<vector<vector<int>>> img(height);

        for(int i = 0; i < height; i++)
        {
            img[height - i - 1] = vector<vector<int>>(width);

            for(int j = 0; j < width; j++)
            {
                auto ray = pixelRay(j, i, driver, normalizedCameraWVector, normalizedCameraUVector, cameraVVector);

                Eigen::Vector3d color = findNearestSphere(driver, ray);

                vector<int> intColor(3);
                intColor[0] = static_cast<int>(color[0] * 255);
                intColor[1] = static_cast<int>(color[1] * 255);
                intColor[2] = static_cast<int>(color[2] * 255);

                img[height - i - 1][j] = intColor;

            }
        }

        // Initialize Object Writer Object. This will create and write to the correctly named file.

        PWriter writer(argv[2]);

        writer << img;
    }

    catch(invalid_argument &err)
    {
        cerr << err.what();
    }
}

