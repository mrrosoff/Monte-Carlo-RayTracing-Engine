//
// Created by Max Rosoff on 9/7/2019.
//

#include "DReader.h"

using namespace std;

DReader &DReader::operator<<(const string &file)
{
    this->driverFile = file;
    this->driverName = findDriverName(file);
    readDriver(file);
    return *this;
}

Eigen::Vector3d DReader::parseEye(const vector<string> &lineData) const
{
    return {stod(lineData[1]),stod(lineData[2]), stod(lineData[3])};
}

Eigen::Vector3d DReader::parseLook(const vector<string> &lineData) const
{
    return {stod(lineData[1]),stod(lineData[2]), stod(lineData[3])};
}

Eigen::Vector3d DReader::parseUp(const vector<string> &lineData) const
{
    return {stod(lineData[1]),stod(lineData[2]), stod(lineData[3])};
}

double DReader::parseD(const vector<string> &lineData) const
{
    return stod(lineData[1]);
}

Eigen::Vector4d DReader::parseBounds(const vector<string> &lineData) const
{
    return {stod(lineData[1]),stod(lineData[2]), stod(lineData[3]), stod(lineData[4])};
}

Eigen::Vector2d DReader::parseRes(const vector<string> &lineData) const
{
    return {stod(lineData[1]),stod(lineData[2])};
}

Eigen::Vector3d DReader::parseAmbient(const vector<string> &lineData) const
{
    return {stod(lineData[1]),stod(lineData[2]), stod(lineData[3])};
}

void DReader::parseLight(const vector<string> &lineData)
{
    Eigen::Vector3d position;
    Eigen::Vector3d rgb;
    double w;

    position << stod(lineData[1]),
                stod(lineData[2]),
                stod(lineData[3]);

    w = stod(lineData[4]);

    rgb << stod(lineData[5]),
           stod(lineData[6]),
           stod(lineData[7]);

    lights.emplace_back(position, rgb, w);
}

void DReader::parseSphere(const vector<string> &lineData)
{
    Eigen::Vector3d position;
    double radius;
    Eigen::Vector3d Ka;
    Eigen::Vector3d Kd;
    Eigen::Vector3d Ks;
    Eigen::Vector3d Kr;

    position << stod(lineData[1]),
                stod(lineData[2]),
                stod(lineData[3]);

    radius = stod(lineData[4]);

    Ka << stod(lineData[5]),
          stod(lineData[6]),
          stod(lineData[7]);

    Kd << stod(lineData[8]),
          stod(lineData[9]),
          stod(lineData[10]);

    Ks << stod(lineData[11]),
          stod(lineData[12]),
          stod(lineData[13]);

    Kr << stod(lineData[14]),
          stod(lineData[15]),
          stod(lineData[16]);

    Material mat("A Sphere Material", Ka, Kd, Ks, Kr);
    spheres.emplace_back(position, radius, mat);
}

int DReader::parseRecursionLevel(const vector<string> &lineData)
{
    return stoi(lineData[1]);
}

void DReader::parseModel(const vector<string> &lineData)
{
    Eigen::Vector3d rotationVector;

    rotationVector << stod(lineData[1]),
                      stod(lineData[2]),
                      stod(lineData[3]);

    double theta = stod(lineData[4]);

    Eigen::Matrix4d scalar;
    scalar << stod(lineData[5]), 0, 0, 0,
              0, stod(lineData[5]), 0, 0,
              0, 0, stod(lineData[5]), 0,
              0, 0, 0, 1;

    Eigen::Matrix4d translation;
    translation << 1, 0, 0, stod(lineData[6]),
                   0, 1, 0, stod(lineData[7]),
                   0, 0, 1, stod(lineData[8]),
                   0, 0, 0, 1;

    string modelPath = lineData[9];

    Remap map(rotationVector, theta, scalar, translation, modelPath);
    objs.emplace_back(map);
}

void DReader::readDriver(const string &file)
{
    ifstream driverReader(file);

    if (!driverReader)
    {
        string err = strerror(errno);
        throw invalid_argument("Failure to open Driver File - " + file + ": " + err);
    }

    Eigen::Vector3d eye;
    Eigen::Vector3d look;
    Eigen::Vector3d up;
    double focLen = 0;
    Eigen::Vector4d bounds;
    Eigen::Vector2d res;

    string driverLine;

    while(getline(driverReader, driverLine))
    {
        stringstream lineReader(driverLine);
        string token;

        vector<string> lineData;

        while(getline(lineReader, token, ' '))
        {
            if(!token.empty())
            {
                lineData.push_back(token);
            }
        }

        if(lineData[0] == "eye")
        {
            eye = parseEye(lineData);
        }

        else if(lineData[0] == "look")
        {
            look = parseLook(lineData);
        }

        else if(lineData[0] == "up")
        {
             up = parseUp(lineData);
        }

        else if(lineData[0] == "d")
        {
            focLen = parseD(lineData);
        }

        else if(lineData[0] == "bounds")
        {
            bounds = parseBounds(lineData);
        }

        else if(lineData[0] == "res")
        {
            res = parseRes(lineData);
        }

        else if(lineData[0] == "ambient")
        {
            ambientLight = parseAmbient(lineData);
        }

        else if(lineData[0] == "light")
        {
            parseLight(lineData);
        }

        else if(lineData[0] == "sphere")
        {
            parseSphere(lineData);
        }

        else if(lineData[0] == "recursionlevel")
        {
            recursionDepth = parseRecursionLevel(lineData);
        }

        else if(lineData[0] == "model")
        {
            parseModel(lineData);
        }

        else if(lineData[0] == "#" || lineData[0][0] == '#')
        {
            continue;
        }

        else
        {
            throw invalid_argument("Invalid Driver File Argument, File - " + file);
        }
    }

    camera = Camera(eye, look, up, bounds, focLen, res);
}

string DReader::findDriverName(const string &file)
{
    unsigned long pIndex = driverFile.find_last_of('.');
    unsigned long sIndex = driverFile.find_last_of('/');
    sIndex += 1;
    return driverFile.substr(sIndex, pIndex - sIndex);
}

ostream &operator<<(ostream &out, const DReader &driver)
{
    Eigen::IOFormat ArrayFormat(Eigen::StreamPrecision, 0, "", ", ", "", "", "[", "]");

    cout << '\n' << "Scene Setup" << '\n' << "-----------" << '\n';

    out << "Driver File: " << driver.driverFile << '\n';
    out << "Driver FileName: " << driver.driverName << '\n';
    out << "Driver Camera: " << driver.camera << '\n';
    out << "Driver Depth: " << driver.recursionDepth << '\n';
    out << "Driver Ambient Light: " << driver.ambientLight.format(ArrayFormat) << '\n';

    cout << '\n' << "Scene Items" << '\n' << "-----------" << '\n';

    cout << '\n' << "Lights" << '\n' << "------" << '\n';
    for(size_t i = 0; i < driver.lights.size(); i++)
    {
        out << "Light " << i << ": " << driver.lights[i] << '\n';
    }

    cout << '\n' << "Spheres" << '\n' << "-------" << '\n';
    for(size_t i = 0; i < driver.spheres.size(); i++)
    {
        out << "Sphere " << i << ": " << driver.spheres[i] << '\n';
    }

    cout << '\n' << "Objects" << '\n' << "------" << '\n';
    for(size_t i = 0; i < driver.objs.size(); i++)
    {
        out << "Objects " << i << ": " << driver.objs[i] << '\n';
    }

    return out;
}
