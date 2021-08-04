//
// Created by Max Rosoff on 9/21/2019.
//

#include <iostream>

#include "RayTracing/RayTracer.h"

using namespace std;

int main(int argc, char** argv)
{
    if(argc != 4)
    {
        cerr << "Usage: ./raytracer" << " " << "[Driver File]" << " " << "[PPM Output File]" << " " << "[Samples Per Pixel]" << '\n';
        return 1;
    }
    
    return RayTracer(argv).rayTrace();
}