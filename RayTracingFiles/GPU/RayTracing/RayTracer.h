//
// Created by Max Rosoff on 11/11/2019.
//

#ifndef RAYTRACER_RAYTRACER_H
#define RAYTRACER_RAYTRACER_H

#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <random>

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

    explicit RayTracer(const DReader &, int, std::default_random_engine, std::uniform_real_distribution<double>);
    __device__ Vector calculateColor(Ray &, Vector, int);

    int samples;
    DReader driver;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution;

private:

    __device__ Vector makeRandomUnitVector();
    __device__ bool checkForIntersection(Ray &) const;
};


#endif //RAYTRACER_RAYTRACER_H
