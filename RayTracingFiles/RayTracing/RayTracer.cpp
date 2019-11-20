//
// Created by Max Rosoff on 9/21/2019.
//

#include "RayTracer.h"

using namespace std::chrono;
using namespace std;

RayTracer::RayTracer(char** argv, const bool isMonteCarlo, const int samples) :

inFile(argv[1]), outFile(argv[2]), isMonteCarlo(isMonteCarlo), samples(samples)

{}

Eigen::Vector3d RayTracer::makeRandomUnitVector()
{
    while(true)
    {
        Eigen::Vector3d returnVector;

        default_random_engine generator(system_clock::now().time_since_epoch().count());
        uniform_real_distribution<double> distribution(-1,1);

        for (int i = 0; i < 3; i++)
        {
            returnVector[i] = distribution(generator);
        }

        returnVector.normalize();

        if(returnVector.dot(returnVector))
        {
            return returnVector;
        }
    }

    return {0, 0, 0};
}

int RayTracer::rayTrace() {

    try
    {
        cout << "Reading All Files." << endl;

        driver << isMonteCarlo << inFile;

        //cout << driver << endl; // Uncomment for Debugging.

        auto resolution = driver.camera.resolution;

        int width = resolution[0];
        int height = resolution[1];

        vector<vector<vector<int>>> img(height);

        cout << "Beginning Raytracing." << endl;

        auto start = high_resolution_clock::now();
        int counter = 10;

        #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(dynamic)
        for(int i = 0; i < height; i++)
        {
            img[i] = vector<vector<int>>(width);

            for(int j = 0; j < width; j++)
            {
                img[i][j] = vector<int>(3);

                auto ray = driver.camera.pixelRay(j, i);

                Eigen::Vector3d color;
                Eigen::Vector3d startWithFullReflect = {1, 1, 1};

                if(isMonteCarlo)
                {
                    color = {0, 0, 0};

                    for(int sample = 0; sample < samples + 1; sample++)
                    {
                        color += calculateMCColor(ray, startWithFullReflect, driver.recursionDepth);

                        if(sample < samples)
                        {
                            ray = driver.camera.pixelRay(j, i);
                        }
                    }

                    for(int i = 0; i < 3; i++)
                    {
                        color[i] = sqrt(color[i] / samples);
                    }
                }

                else
                {
                    color = calculateColor(ray, startWithFullReflect, driver.recursionDepth);
                }

                img[i][j][0] = min(max(static_cast<int>(color[0] * 255), 0), 255);
                img[i][j][1] = min(max(static_cast<int>(color[1] * 255), 0), 255);
                img[i][j][2] = min(max(static_cast<int>(color[2] * 255), 0), 255);
            }

            int percentComplete = static_cast<int>(floor((static_cast<double>(i) / height) * 100));

            #pragma omp critical
            if(percentComplete == counter)
            {
                cout << percentComplete << "% complete." << endl;
                counter += 10;
            }
        }

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        double durationCount = duration.count();

        cout << "Raytracer ran in " << durationCount * 0.000001 << " seconds." << endl;
        cout << "Writing to PPM File." << '\n';

        PWriter writer(outFile);
        writer << img;

        cout << "Finished!" << '\n';
        return 0;
    }

    catch(invalid_argument &err)
    {
        cerr << "\033[1;31m" << err.what() << "\033[0m" << endl;
        return 1;
    }
}

Eigen::Vector3d RayTracer::calculateColor(Ray &ray, const Eigen::Vector3d &howMuchReflect, const int depth) {

    if (!checkForIntersection(ray))
    {
        return {0, 0, 0};
    }

    Eigen::Vector3d calculatedColor = calculateTraditionalColor(ray, howMuchReflect);

    if (depth > 0)
    {
        auto invDirection = -1 * ray.direction;
        Eigen::Vector3d reflectionDirection = (2 * ray.surfaceNormal.dot(invDirection) * ray.surfaceNormal - invDirection).normalized();

        auto newRay = Ray(ray.closestIntersectionPoint, reflectionDirection);
        calculatedColor += calculateColor(newRay, ray.material.Kr.cwiseProduct(howMuchReflect), depth - 1);
    }


    if(depth > 0 && ray.material.illum == 6)
    {
        try
        {
            Ray invRay(ray.closestIntersectionPoint, -1 * ray.direction);
            invRay.surfaceNormal = ray.surfaceNormal;
            Ray refractedRay = ray.hit->makeExitRefrationRay(invRay, ray.material.Ni, 1.0);
            calculatedColor += calculateColor(refractedRay, ray.material.Ko.cwiseProduct(howMuchReflect), depth - 1);
        }

        catch(const range_error &err) {}
    }

    return calculatedColor;
}

Eigen::Vector3d RayTracer::calculateTraditionalColor(const Ray &ray, const Eigen::Vector3d &howMuchReflect)
{
    Eigen::Vector3d ambientColor = driver.ambientLight.cwiseProduct(ray.material.Ka);

    for (const auto &light : driver.lights)
    {
        auto lightVector = (light.position - ray.closestIntersectionPoint).normalized();
        auto nSNDotLV = ray.surfaceNormal.dot(lightVector);

        if (nSNDotLV > 0)
        {
            auto shadowRayDirection = light.position - ray.closestIntersectionPoint;
            auto shadowRay = Ray(ray.closestIntersectionPoint, shadowRayDirection, shadowRayDirection.norm());

            if (checkForIntersection(shadowRay, true))
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

    return howMuchReflect.cwiseProduct(ambientColor);
}

Eigen::Vector3d RayTracer::calculateMCColor(Ray &ray, const Eigen::Vector3d currentAlbedo, const int depth)
{
    if (!checkForIntersection(ray))
    {
        return {0, 0, 0};
    }

    else if(ray.hit->isLight)
    {
        return currentAlbedo.cwiseProduct(ray.material.Kd);
    }

    else if (depth > 0)
    {
	Ray newRay;

	if(ray.material.Kr[0] > 0 || ray.material.Kr[1] > 0 || ray.material.Kr[2] > 0)
	{
            auto invDirection = -1 * ray.direction;
            Eigen::Vector3d reflectionDirection = (2 * ray.surfaceNormal.dot(invDirection) * ray.surfaceNormal - invDirection).normalized();

            newRay = Ray(ray.closestIntersectionPoint, reflectionDirection);
	}

        else
	{
	    newRay = Ray(ray.closestIntersectionPoint, makeRandomUnitVector());
	}

        return calculateMCColor(newRay, currentAlbedo.cwiseProduct(ray.material.Kd), depth - 1);
    }

    return currentAlbedo;
}

bool RayTracer::checkForIntersection(Ray &ray, const bool isShadow)
{
    bool foundItem = false;

    for(const auto &item : driver.items)
    {
        if(item->intersectionTest(ray))
        {
            if(isShadow)
            {
                return true;
            }

            foundItem = true;
        }
    }

    return foundItem;
}
