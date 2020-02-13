//
// Created by Max Rosoff on 11/11/2019.
//

#ifndef RAYTRACER_RAYTRACER_H
#define RAYTRACER_RAYTRACER_H

#include <iostream>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <random>
#include <iomanip>

#include "Ray.h"
#include "../Matrix/Vector.h"
#include "../FileInOut/DReader.h"
#include "../FileInOut/PWriter.h"

class RayTracer {

public:

    RayTracer() = delete;
    RayTracer(const RayTracer &) = default;
    RayTracer &operator=(const RayTracer &) = delete;
    ~RayTracer() = default;

    explicit RayTracer(char**);
    int rayTrace();

private:

    Vector makeRandomUnitVector();
    Vector calculateAverageColor(int, int);
    Vector calculateColor(Ray &, Vector, int);
    bool checkForIntersection(Ray &);

    DReader driver;
    std::string inFile;
    std::string outFile;
    int samples = 0;

    // Random Number Generator
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution;
};


#endif //RAYTRACER_RAYTRACER_H
