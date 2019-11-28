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

                Eigen::Vector3d color = {0, 0, 0};

                for(int k = 0; k < samples + 1; k++)
                {
                    color += calculateColor(ray, {1, 1, 1}, 50);

                    if(k < samples)
                    {
                        ray = driver.camera.pixelRay(j, i);
                    }
                }

                for(int k = 0; k < 3; k++)
                {
                    img[i][j][k] = min(max(static_cast<int>(sqrt(color[k] / samples) * 255), 0), 255);
                }
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
            auto invDirection = -1 * ray.direction;
            Eigen::Vector3d reflectionDirection = (2 * ray.surfaceNormal.dot(invDirection) * ray.surfaceNormal - invDirection).normalized();

            newRay = Ray(ray.closestIntersectionPoint, reflectionDirection);
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
