//
// Created by Max Rosoff on 9/21/2019.
//

#include <iostream>

#include "RayTracingFiles/RayTracing/RayTracer.h"

using namespace std;

int main(int argc, char** argv)
{
    if(argc < 3 || argc > 5)
    {
        cerr << "Usage: ./raytracer" << " " << "[Driver File]" << " " << "[PPM Output File]" << '\n';
        return 1;
    }

    bool isMonteCarlo = false;
    int samples = 0;

    if(argc == 4)
    {
        isMonteCarlo = true;
        samples = stoi(argv[3]);
    }

    return RayTracer(argv, isMonteCarlo, samples).rayTrace();
}
