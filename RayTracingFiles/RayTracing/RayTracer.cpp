//
// Created by Max Rosoff on 9/21/2019.
//

#include "RayTracer.h"

using namespace std::chrono;
using namespace std;

RayTracer::RayTracer(char** argv) :

inFile(argv[1]), outFile(argv[2]), samples(stoi(argv[3]))

{}

Eigen::Vector3d RayTracer::makeRandomUnitVector()
{
    Eigen::Vector3d returnVector;

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

int RayTracer::rayTrace() {

    try
    {
        cout << "Reading All Files." << endl;

        driver << inFile;

        //cout << driver << endl; // Uncomment for Debugging.

        auto resolution = driver.camera.resolution;

        int width = resolution[0];
        int height = resolution[1];

        vector<vector<vector<int>>> img(height);

        cout << "Beginning Ray Tracing." << endl;

        auto start = high_resolution_clock::now();
        double currentPercentCompleted = 0;

        // Instantiate Random Number Generator
        generator = default_random_engine(system_clock::now().time_since_epoch().count());
        distribution = uniform_real_distribution<double>(-1,1);

        #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(dynamic)
        for(int i = 0; i < height; i++)
        {
            img[i] = vector<vector<int>>(width);

            for(int j = 0; j < width; j++)
            {
                img[i][j] = vector<int>(3);

                Eigen::Vector3d color = calculateAverageColor(i, j);

                for(int k = 0; k < 3; k++)
                {
                    img[i][j][k] = min(max(static_cast<int>(sqrt(color[k] / samples) * 255), 0), 255);
                }
            }

            auto percentComplete = (static_cast<double>(i) / height) * 100;

            #pragma omp critical
            if(percentComplete > currentPercentCompleted)
            {
                currentPercentCompleted = percentComplete;

                auto currentTime = high_resolution_clock::now();
                auto durationInSeconds = duration_cast<seconds>(currentTime - start).count();
                auto timeRemaining = (durationInSeconds / percentComplete) * (100 - percentComplete);

                cout << '\r' << static_cast<int>(floor(percentComplete)) << "% complete.";
                cout << " Estimated Time Remaining: " << timeRemaining << " seconds. " << flush;
            }
        }

        cout << '\r';

        auto stop = high_resolution_clock::now();
        auto durationInSeconds = duration_cast<seconds>(stop - start).count();

        cout << "Ray Tracer ran in " << durationInSeconds << " seconds." << endl;
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

Eigen::Vector3d RayTracer::calculateAverageColor(const int i, const int j)
{
    auto theRay = driver.camera.pixelRay(j, i);
    auto loopedRay = theRay;

    Eigen::Vector3d color = {0, 0, 0};

    for(int k = 0; k < samples; k++)
    {
        color += calculateColor(loopedRay, {1, 1, 1}, 30);
        loopedRay = theRay;
    }

    return color + calculateColor(loopedRay, {1, 1, 1}, 30);
}

Eigen::Vector3d RayTracer::calculateColor(Ray &ray, const Eigen::Vector3d &currentAlbedo, const int depth)
{
    if (!checkForIntersection(ray))
    {
        return {0, 0, 0};
    }

    else if(ray.material.isLight)
    {
        return currentAlbedo.cwiseProduct(ray.material.albedo);
    }

    else if (depth > 0)
    {
        Ray newRay;

        if(ray.material.isMirror)
        {
            Eigen::Vector3d reflectionDirection = (2 * ray.surfaceNormal.dot(-1 * ray.direction) * ray.surfaceNormal + ray.direction).normalized();
            newRay = Ray(ray.closestIntersectionPoint, reflectionDirection + 0.1 * makeRandomUnitVector());
        }

        else if(ray.material.isGlass)
        {
            try
            {
                newRay = ray.hit->makeExitRefrationRay(ray, 1.5, 1.0);
            }

            catch(const range_error &error) {}
        }

        else
        {
            newRay = Ray(ray.closestIntersectionPoint, ray.surfaceNormal + makeRandomUnitVector());
        }

        return calculateColor(newRay, currentAlbedo.cwiseProduct(ray.material.albedo), depth - 1);
    }

    return currentAlbedo;
}

bool RayTracer::checkForIntersection(Ray &ray)
{
    for(const auto &item : driver.items)
    {
        item->intersectionTest(ray);
    }

    return ray.hit != nullptr;
}
