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
#include "../FileInOut/DReader.h"
#include "../FileInOut/PWriter.h"

#include "../../Eigen/Eigen"

class RayTracer {

public:

    RayTracer() = delete;
    RayTracer(const RayTracer &) = default;
    RayTracer &operator=(const RayTracer &) = delete;
    ~RayTracer() = default;

    explicit RayTracer(char**, bool = false, int = 0);
    int rayTrace();

private:

    static Eigen::Vector3d makeRandomUnitVector();
    Eigen::Vector3d calculateColor(Ray &, const Eigen::Vector3d &, int);
    Eigen::Vector3d calculateMCColor(Ray &, const Eigen::Vector3d, int);
    Eigen::Vector3d calculateTraditionalColor(const Ray &, const Eigen::Vector3d &);
    bool checkForIntersection(Ray &, bool isShadow = false);

    DReader driver;
    std::string inFile;
    std:: string outFile;
    bool isMonteCarlo;
    int samples;
};


#endif //RAYTRACER_RAYTRACER_H
