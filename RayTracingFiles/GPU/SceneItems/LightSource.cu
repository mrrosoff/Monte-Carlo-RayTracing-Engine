//
// Created by Max Rosoff on 9/21/2019.
//

#include "LightSource.h"

using namespace std;

LightSource::LightSource(const Vector &position, const Vector &rgb, const double w) :

position(position), rgb(rgb), w(w)

{}

ostream &operator<<(ostream &out, const LightSource &light)
{

    out << "Light Position: " << light.position;
    out << "Light RGB: " << light.rgb;
    out << "Light W: " << light.w << '\n';

    return out;
}