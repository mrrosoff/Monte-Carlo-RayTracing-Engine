//
// Created by Max Rosoff on 9/21/2019.
//

#include "RayTracer.h"

using namespace std::chrono;
using namespace std;

#ifndef NODE
RayTracer::RayTracer(char** argv) :

inFile(argv[1]), outFile(argv[2]), samples(stoi(argv[3]))

{}
#endif

Vector RayTracer::makeRandomUnitVector()
{
    Vector returnVector(3);

    while (true)
    {
        for (int i = 0; i < 3; i++)
        {
            returnVector[i] = distribution(generator);
        }

        returnVector.normalize();

        if (returnVector.dot(returnVector) < 1)
        {
            return returnVector;
        }
    }
}

#ifdef NODE
string RayTracer::rayTrace(const Napi::Env &env, const Napi::Function &emit, const vector<string> &sceneData, const vector<string> &spheres)
{

    try
    {
        string file = "eye " + sceneData[0] + '\n' +
                      "look " + sceneData[1] + '\n' +
                      "up " + sceneData[2] + '\n' +
                      "d " + sceneData[3] + '\n' +
                      "bounds " + sceneData[4] + '\n' +
                      "res " + sceneData[5] + '\n';

        for (const auto &sphere : spheres)
        {
            file += "sphere " + sphere + '\n';
        }

        emit.Call({Napi::String::New(env, "start"), Napi::String::New(env, file)});

        driver << file;

        // cout << driver << endl; // Uncomment for Debugging.

        samples = stoi(sceneData[6]);

        auto resolution = driver.camera.resolution;

        int width = resolution[0];
        int height = resolution[1];

        vector<vector<vector<int>>> img(height);

        auto start = high_resolution_clock::now();
        double currentPercentCompleted = 0;

        // Instantiate Random Number Generator
        generator = default_random_engine(system_clock::now().time_since_epoch().count());
        distribution = uniform_real_distribution<double>(-1, 1);

        for (int i = 0; i < height; i++)
        {
            img[i] = vector<vector<int>>(width);

            for (int j = 0; j < width; j++)
            {
                img[i][j] = vector<int>(3);

                Vector color = calculateAverageColor(i, j);

                for (int k = 0; k < 3; k++)
                {
                    img[i][j][k] = min(max(static_cast<int>(sqrt(color[k] / samples) * 255), 0), 255);
                }
            }

            auto percentComplete = (static_cast<double>(i) / height) * 100;

            if (percentComplete > currentPercentCompleted)
            {
                currentPercentCompleted = percentComplete;

                auto currentTime = high_resolution_clock::now();
                auto durationInSeconds = duration_cast<seconds>(currentTime - start).count();
                auto timeRemaining = (durationInSeconds / percentComplete) * (100 - percentComplete);

                string stringPercent;
                string stringTime;
                stringstream stringPercentStream;
                stringstream stringTimeStream;

                stringPercentStream << static_cast<int>(floor(percentComplete)) << "% complete.";
                stringTimeStream << "Estimated Time Remaining: " << timeRemaining << " seconds. " << flush;

                getline(stringPercentStream, stringPercent);
                getline(stringTimeStream, stringTime);

                emit.Call({Napi::String::New(env, "progress"), Napi::String::New(env, stringPercent), Napi::String::New(env, stringTime)});
            }
        }

        auto stop = high_resolution_clock::now();
        auto durationInSeconds = duration_cast<seconds>(stop - start).count();

        string stringTimeToRun;
        stringstream stringTimeToRunStream;

        stringTimeToRunStream << "Raytracer ran in " << durationInSeconds << " seconds.";

        getline(stringTimeToRunStream, stringTimeToRun);

        emit.Call({Napi::String::New(env, "finish"), Napi::String::New(env, stringTimeToRun)});

        stringstream ss;
        string output;

        ss << "P3" << '\n'
           << width << " " << height << " "
           << "255" << '\n';

        for (size_t i = 0; i < img.size(); i++)
        {
            for (size_t j = 0; j < img[i].size(); j++)
            {
                for (size_t k = 0; k < img[i][j].size(); k++)
                {
                    ss << img[i][j][k];

                    if (k < img[i][j].size() - 1)
                    {
                        ss << " ";
                    }
                }

                if (j < img[i].size() - 1)
                {
                    ss << " ";
                }
            }

            ss << '\n';
        }

        string middleString;

        while (getline(ss, middleString))
        {
            output += middleString + '\n';
        }

        return output;
    }

    catch (invalid_argument &err)
    {
        cerr << "\033[1;31m" << err.what() << "\033[0m" << endl;
        return "";
    }
}

#else

int RayTracer::rayTrace()
{

    try
    {
        cout << "Reading All Files." << endl;

        driver << inFile;

        // cout << driver << endl; // Uncomment for Debugging.

        auto resolution = driver.camera.resolution;

        int width = resolution[0];
        int height = resolution[1];

        vector<vector<vector<int>>> img(height);

        cout << "Beginning Ray Tracing. Num Threads: " << omp_get_max_threads() << endl;

        auto start = high_resolution_clock::now();
        double currentPercentCompleted = 0;

        // Instantiate Random Number Generator
        generator = default_random_engine(system_clock::now().time_since_epoch().count());
        distribution = uniform_real_distribution<double>(-1, 1);

        #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(dynamic)
        for (int i = 0; i < height; i++)
        {
            img[i] = vector<vector<int>>(width);

            for (int j = 0; j < width; j++)
            {
                img[i][j] = vector<int>(3);

                Vector color = calculateAverageColor(i, j);

                for (int k = 0; k < 3; k++)
                {
                    img[i][j][k] = min(max(static_cast<int>(sqrt(color[k] / samples) * 255), 0), 255);
                }
            }

            auto percentComplete = (static_cast<double>(i) / height) * 100;

            #pragma omp critical
            if (percentComplete > currentPercentCompleted)
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

        cout << "100% Complete. Ray Tracer ran in " << durationInSeconds << " seconds. Preparing to Output Image." << endl;
        cout << "Writing to PPM File." << '\n';

        PWriter writer(outFile);
        writer << img;

        cout << "Finished!" << '\n';
        return 0;
    }

    catch (invalid_argument &err)
    {
        cerr << "\033[1;31m" << err.what() << "\033[0m" << endl;
        return 1;
    }
}
#endif

Vector RayTracer::calculateAverageColor(const int i, const int j)
{
    auto theRay = driver.camera.pixelRay(j, i);
    auto loopedRay = theRay;

    Vector color = {0, 0, 0};

    for (int k = 0; k < samples - 1; k++)
    {
        color += calculateColor(loopedRay, {1, 1, 1}, 10);
        loopedRay = theRay;
    }

    return color + calculateColor(loopedRay, {1, 1, 1}, 10);
}

Vector RayTracer::calculateColor(Ray &ray, Vector currentAlbedo, const int depth)
{
    double max = 0;

    for (int i = 0; i < 3; i++)
    {
        if (currentAlbedo[i] > max)
        {
            max = currentAlbedo[i];
        }
    }

    if (max < abs(distribution(generator)))
    {
        return {0, 0, 0};
    }

    currentAlbedo /= max;

    if (!checkForIntersection(ray))
    {
        return {0, 0, 0};
    }

    else if (ray.material.isLight)
    {
        return currentAlbedo * ray.material.albedo;
    }

    else if (depth > 0)
    {
        Ray newRay;

        if (ray.material.isMirror)
        {
            Vector reflectionDirection = (2 * ray.surfaceNormal.dot(-ray.direction) * ray.surfaceNormal + ray.direction).normalize();
            newRay = Ray(ray.closestIntersectionPoint, reflectionDirection + 0.02 * makeRandomUnitVector());
        }

        else if (ray.material.isGlass)
        {
            try
            {
                newRay = ray.hit->makeExitRefrationRay(ray, 1.0, 1.5);
            }

            catch (const range_error &error)
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

bool RayTracer::checkForIntersection(Ray &ray) const
{
    for (const auto &item : driver.items)
    {
        item->intersectionTest(ray);
    }

    return ray.hit != nullptr;
}
