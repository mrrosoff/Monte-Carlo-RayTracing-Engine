//
// Created by Max Rosoff on 9/21/2019.
//

#include "RayTracer.h"

using namespace std;


__host__ RayTracer::RayTracer(const DReader &, int samples, std::default_random_engine generator, std::uniform_real_distribution<double> distribution)

: driver(driver), samples(samples), generator(generator), distribution(distribution)

{}

__device__ Vector<3> RayTracer::makeRandomUnitVector()
{
    Vector<3> returnVector;

    while(true)
    {
        for(int i = 0; i < 3; i++)
        {
            returnVector[i] = distribution(generator);
        }

        returnVector.normalize();

        if(returnVector.dot(returnVector) < 1)
        {
            return returnVector;
        }
    }
}

__device__ Vector<3> RayTracer::calculateColor(Ray &ray, Vector<3> currentAlbedo, const int depth)
{
    double max = 0;

    for(int i = 0; i < 3; i++)
    {
        if(currentAlbedo[i] > max)
        {
            max = currentAlbedo[i];
        }
    }

    if(max < abs(distribution(generator)))
    {
        return {0, 0, 0};
    }

    currentAlbedo /= max;

    if (!checkForIntersection(ray))
    {
        return {0, 0, 0};
    }

    else if(ray.material.isLight)
    {
        return currentAlbedo * ray.material.albedo;
    }

    else if (depth > 0)
    {
        Ray newRay;

        if(ray.material.isMirror)
        {
            Vector reflectionDirection = (2 * ray.surfaceNormal.dot(-ray.direction) * ray.surfaceNormal + ray.direction).normalize();
            newRay = Ray(ray.closestIntersectionPoint, reflectionDirection + 0.02 * makeRandomUnitVector());
        }

        else if(ray.material.isGlass)
        {
            newRay = ray.hit->makeExitRefrationRay(ray, 1.0, 1.5);

            if(newRay[0] == 0 && newRay[1] == 0 && newRay[2] == 0)
            {
                newRay = Ray(ray.closestIntersectionPoint, ray.surfaceNormal + makeRandomUnitVector());
            }
        }

        else
        {
            newRay = Ray(ray.closestIntersectionPoint, ray.surfaceNormal + makeRandomUnitVector());
        }

        return calculateColor(newRay, currentAlbedo * ray.material.albedo, depth - 1);
    }

    return currentAlbedo;
}

__device__ bool RayTracer::checkForIntersection(Ray &ray) const
{
    for(const auto &item : driver.items)
    {
        item->intersectionTest(ray);
    }

    return ray.hit != nullptr;
}
