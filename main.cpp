//
// Created by Max Rosoff on 9/21/2019.
//

#include <iostream>
#include "RayTracingFiles/RayTracing/RayTracer.h"

using namespace std;

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        cerr << "Usage: ./raytracer" << " " << "[Driver File]" << " " << "[PPM Output File]" << '\n';
        return 1;
    }

    RayTracer(argv).rayTrace();
    return 0;
}
