//
// Created by Max Rosoff on 9/21/2019.
//

#include <iostream>
#include <chrono>

#include "RayTracing/RayTracer.h"

using namespace std;
using namespace std::chrono;

__host__ int rayTrace(const char** argv);
__global__ void calculateAverageColor(double *frameBuffer, const int maxX, const int maxY);

__host__ int main(const int argc, const char** argv)
{
    if(argc != 4)
    {
        cerr << "Usage: ./raytracer" << " " << "[Driver File]" << " " << "[PPM Output File]" << " " << "[Samples Per Pixel]" << '\n';
        return 1;
    }

    return rayTrace(argv);
}


__host__ int rayTrace(const char** argv) {

    try
    {
        cout << "Reading All Files." << endl;

        DReader driver(argv[1]);

        auto resolution = driver.camera.resolution;

        int width = resolution[0];
        int height = resolution[1];
        int samples = stoi(argv[3]);

        // Instantiate Random Number Generator

        std::default_random_engine generator = default_random_engine(system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<double> distribution = uniform_real_distribution<double>(-1, 1);

        RayTracer(driver, samples, generator, distribution);

        vector<vector<vector<int>>> img(height);

        cout << "Beginning Ray Tracing." << endl;

        auto start = high_resolution_clock::now();

        int numPixels = width * height;
        size_t frameBufferSize = 3 * numPixels * sizeof(double);

        double *frameBuffer;
        cudaMallocManaged((void **) &frameBuffer, frameBufferSize);

        int xThreads = 8, yThreads = 8;

        dim3 blocks(width / xThreads + 1, height / yThreads + 1);
        dim3 threads(xThreads, yThreads);
        calculateAverageColor<<<blocks, threads>>>(frameBuffer, width, height);

        cudaGetLastError();
        cudaDeviceSynchronize();

        for(int i = 0; i < height; i++)
        {
            img[i] = vector<vector<int>>(width);

            for(int j = 0; j < width; j++)
            {
                img[i][j] = vector<int>(3);

                size_t pixelIndex = i * 3 * width + j * 3;

                for(int k = 0; k < 3; k++)
                {
                    img[i][j][k] = min(max(static_cast<int>(sqrt(frameBuffer[pixelIndex + k] / samples) * 255), 0), 255);
                }
            }
        }

        cudaFree(frameBuffer);

        auto stop = high_resolution_clock::now();
        auto durationInSeconds = duration_cast<seconds>(stop - start).count();

        cout << "Ray Tracer ran in " << durationInSeconds << " seconds. Preparing to Output Image." << endl;
        cout << "Writing to PPM File." << '\n';

        PWriter writer(argv[2]);
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


__global__ void calculateAverageColor(const RayTracer &trace, double *frameBuffer, const int maxX, const int maxY)
{
    // Identify Pixel Location (i, j)

    double i = threadIdx.y + blockIdx.y * blockDim.y;
    double j = threadIdx.x + blockIdx.x * blockDim.x;

    if(i >= maxY || j >= maxX)
    {
        return;
    }

    int pixelIndex = i * maxX * 3 + j * 3;

    // Do Ray Trace

    auto theRay = trace.driver.camera.pixelRay(j, i);
    auto loopedRay = theRay;

    Vector color = {0, 0, 0};

    for(int k = 0; k < trace.samples - 1; k++)
    {
        color += trace.calculateColor(loopedRay, {1, 1, 1}, 10);
        loopedRay = theRay;
    }

    color += trace.calculateColor(loopedRay, {1, 1, 1}, 10);

    // Load GPU Frame Buffer

    frameBuffer[pixelIndex + 0] = color[0];
    frameBuffer[pixelIndex + 1] = color[1];
    frameBuffer[pixelIndex + 2] = color[2];
}