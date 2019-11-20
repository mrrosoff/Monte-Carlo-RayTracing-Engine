//
// Created by Max Rosoff on 10/19/2019.
//

#include "Material.h"

using namespace std;

Material::Material(const string &name, const Eigen::Vector3d &Ka, const Eigen::Vector3d &Kd, const Eigen::Vector3d &Ks, const Eigen::Vector3d &Kr, const Eigen::Vector3d &Ko, const double Ns, const double Ni, const int ill) :

name(name), Ka(Ka), Kd(Kd), Ks(Ks), Kr(Kr), Ko(Ko), Ns(Ns), Ni(Ni), illum(ill)

{
    if(Ni > 0)
    {
        illum = 6;
    }
}

ostream &operator<<(ostream &out, const Material &mat)
{
    Eigen::IOFormat ArrayFormat(Eigen::StreamPrecision, 0, "", ", ", "", "", "[", "]");

    out << "Material Name: " << mat.name << '\n';
    out << "Material Ka: " << mat.Ka.format(ArrayFormat) << '\n';
    out << "Material Kd: " << mat.Kd.format(ArrayFormat) << '\n';
    out << "Material Ks: " << mat.Ks.format(ArrayFormat) << '\n';
    out << "Material Kr: " << mat.Kr.format(ArrayFormat) << '\n';
    out << "Material Ko: " << mat.Ko.format(ArrayFormat) << '\n';
    out << "Material Ns: " << mat.Ns << '\n';
    out << "Material Ni: " << mat.Ni << '\n';
    out << "Material Illum: " << mat.illum << '\n';

    return out;
}
