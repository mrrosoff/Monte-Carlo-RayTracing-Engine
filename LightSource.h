//
// Created by Max Rosoff on 9/21/2019.
//

#ifndef GRAPHICS_LIGHTSOURCE_H
#define GRAPHICS_LIGHTSOURCE_H

#include "../Eigen/Eigen/Eigen"

class LightSource {

public:

    LightSource() = delete;
    explicit LightSource(const Eigen::Vector3d &, const Eigen::Vector3d &, double);

    LightSource(const LightSource &) = default;
    LightSource &operator=(const LightSource &) = delete;
    ~LightSource() = default;

    Eigen::Vector3d position;
    Eigen::Vector3d rgb;
    double w = 0;

};


#endif //GRAPHICS_LIGHTSOURCE_H
