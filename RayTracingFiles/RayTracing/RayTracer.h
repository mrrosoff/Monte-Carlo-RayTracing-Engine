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

    explicit RayTracer(char**);
    void rayTrace();

private:

    static Eigen::Vector3d doARayTrace(Ray &, const DReader &, const Eigen::Vector3d &, const Eigen::Vector3d &, int);
    static bool checkForIntersection(Ray &, const DReader &, bool isShadow = false);

    std::string inFile;
    std:: string outFile;
};


#endif //RAYTRACER_RAYTRACER_H
