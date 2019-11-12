//
// Created by Max Rosoff on 9/21/2019.
//

#include "RayTracer.h"

using namespace std::chrono;
using namespace std;

RayTracer::RayTracer(char** argv) :

inFile(argv[1]), outFile(argv[2])

{}

void RayTracer::rayTrace() {

    // Initialize Driver Reader Object. This will load public members of the driver file for use.

    try
    {
        DReader driver;

        cout << "Reading All Files." << '\n';

        driver << inFile;

        auto resolution = driver.camera.resolution;

        int width = resolution[0];
        int height = resolution[1];

        vector<vector<vector<int>>> img(height);

        cout << "Beginning Raytracing." << '\n';

        auto start = high_resolution_clock::now();

        #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(dynamic)
        for(int i = 0; i < height; i++)
        {
            img[i] = vector<vector<int>>(width);

            for(int j = 0; j < width; j++)
            {
                img[i][j] = vector<int>(3);

                auto ray = driver.camera.pixelRay(j, i);

                Eigen::Vector3d startWithBlack = {0, 0, 0};
                Eigen::Vector3d startWithFullReflect = {1, 1, 1};

                Eigen::Vector3d color = doARayTrace(ray, driver, startWithBlack, startWithFullReflect, driver.recursionDepth);

                img[i][j][0] = min(max(static_cast<int>(color[0] * 255), 0), 255);
                img[i][j][1] = min(max(static_cast<int>(color[1] * 255), 0), 255);
                img[i][j][2] = min(max(static_cast<int>(color[2] * 255), 0), 255);
            }
        }

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        double durationCount = duration.count();

        cout << "Raytracer ran in " << durationCount * 0.000001 << " seconds." << endl;
        cout << "Writing to PPM File." << '\n';

        // Initialize Object Writer Object. This will create and write to the correctly named file.

        PWriter writer(outFile);
        writer << img;

        cout << "Finished!" << '\n';
    }

    catch(invalid_argument &err)
    {
        cerr << err.what();
    }
}

Eigen::Vector3d RayTracer::doARayTrace(Ray &ray, const DReader &driver, const Eigen::Vector3d &color, const Eigen::Vector3d &howMuchReflect, const int depth)
{
    if(!checkForIntersection(ray, driver))
    {
        return color;
    }

    Eigen::Vector3d ambientColor = driver.ambientLight.cwiseProduct(ray.material.Ka);

    for(const auto &light : driver.lights)
    {
        auto lightVector = (light.position - ray.closestIntersectionPoint).normalized();
        auto nSNDotLV = ray.surfaceNormal.dot(lightVector);

        if (nSNDotLV > 0)
        {
            auto shadowRay = Ray(ray.closestIntersectionPoint, light.position - ray.closestIntersectionPoint);

            if(checkForIntersection(shadowRay, driver, true))
            {
                continue;
            }

            ambientColor += ray.material.Kd.cwiseProduct(light.rgb) * nSNDotLV;

            auto distToQ = (ray.point - ray.closestIntersectionPoint).normalized();
            auto specularReflectAngle = (2 * nSNDotLV * ray.surfaceNormal - lightVector).normalized();
            auto spec = distToQ.dot(specularReflectAngle);

            if (spec > 0)
            {
                ambientColor += ray.material.Ks.cwiseProduct(light.rgb) * pow(spec, ray.material.Ns);
            }
        }
    }

    Eigen::Vector3d newColor = color + howMuchReflect.cwiseProduct(ambientColor);

    if (depth > 0)
    {
        auto invDirection = -1 * ray.direction;
        auto reflectionDirection = (2 * ray.surfaceNormal.dot(invDirection) * ray.surfaceNormal - invDirection).normalized();
        auto newRay = Ray(ray.closestIntersectionPoint, reflectionDirection);
        auto newReflec = ray.material.Kr.cwiseProduct(howMuchReflect);
        return doARayTrace(newRay, driver, newColor, newReflec, depth - 1);
    }

    return newColor;
}

bool RayTracer::checkForIntersection(Ray &ray, const DReader &driver, const bool isShadow)
{
    bool foundItem = false;

    for(const auto &sphere : driver.spheres)
    {
        if(sphere.intersectionTest(ray))
        {
            foundItem = true;

            if(isShadow)
            {
                return foundItem;
            }
        }
    }

    for(const auto &object : driver.objs)
    {
        if(object.intersectionTest(ray))
        {
            foundItem = true;

            if(isShadow)
            {
                return foundItem;
            }
        }
    }

    return foundItem;
}