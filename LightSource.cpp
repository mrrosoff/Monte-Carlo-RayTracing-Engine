//
// Created by Max Rosoff on 9/21/2019.
//

#include "LightSource.h"

using namespace std;

LightSource::LightSource(const Eigen::Vector3d &position, const Eigen::Vector3d &rgb, const double w) :

position(position), rgb(rgb), w(w)

{}

ostream &operator<<(ostream &out, const LightSource &light)
{
    Eigen::IOFormat ArrayFormat(Eigen::StreamPrecision, 0, "", ", ", "", "", "[", "]");

    out << "Light Position: " << light.position.format(ArrayFormat) << '\n';
    out << "Light RGB: " << light.rgb.format(ArrayFormat) << '\n';
    out << "Light W: " << light.w << '\n';

    return out;
}