//
// Created by Max Rosoff on 11/11/2019.
//

#ifndef RAYTRACER_RAYTRACER_H
#define RAYTRACER_RAYTRACER_H

#include <iostream>
#include <iomanip>

#include <string>
#include <sstream>

#include <cmath>
#include <random>
#include <algorithm>

#ifdef NODE
#include <napi.h>
#else
#include <omp.h>
#endif

#include <chrono>

#include "Ray.h"
#include "../Matrix/Vector.h"
#include "../FileInOut/DReader.h"
#include "../FileInOut/PWriter.h"

class RayTracer {

public:

    RayTracer() = default;
    RayTracer(const RayTracer &) = default;
    RayTracer &operator=(const RayTracer &) = delete;
    ~RayTracer() = default;

    #ifdef NODE
    std::string rayTrace(const Napi::Env &, const Napi::Function &, const std::vector<std::string> &, const std::vector<std::string> &);
    #else
    explicit RayTracer(char **);
    int rayTrace();
    #endif

private:

    Vector makeRandomUnitVector();
    Vector calculateAverageColor(int, int);
    Vector calculateColor(Ray &, Vector, int);
    bool checkForIntersection(Ray &) const;

    DReader driver;
    #ifndef NODE
    std::string inFile;
    std::string outFile;
    #endif
    int samples = 0;

    // Random Number Generator
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution;
};

#endif //RAYTRACER_RAYTRACER_H
