//
// Created by Max Rosoff on 9/7/2019.
//

#include "DReader.h"

using namespace std;

DReader &DReader::operator<<(const string &file)
{
    driverFile = file;
    driverName = findDriverName(file);
    readDriver(file);
    return *this;
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

    while (getline(driverReader, driverLine))
    {
        driverLine.erase(remove(driverLine.begin(), driverLine.end(), '\r'), driverLine.end());

        if(driverLine.empty())
        {
            continue;
        }

        stringstream lineReader(driverLine);
        string token;

        vector<string> lineData;

        while (getline(lineReader, token, ' '))
        {
            if (!token.empty())
            {
                lineData.push_back(token);
            }
        }

        if (lineData[0] == "eye")
        {
            throwErrorMessage(lineData.size(), 4, "You must specify all the fields for eye!");
            eye = parseEye(lineData);
        }

        else if (lineData[0] == "look")
        {
            throwErrorMessage(lineData.size(), 4, "You must specify all the fields for look!");
            look = parseLook(lineData);
        }

        else if (lineData[0] == "up")
        {
            throwErrorMessage(lineData.size(), 4, "You must specify all the fields for up!");
            up = parseUp(lineData);
        }

        else if (lineData[0] == "d")
        {
            throwErrorMessage(lineData.size(), 2, "You must specify all the fields for d!");
            focLen = parseD(lineData);
        }

        else if (lineData[0] == "bounds")
        {
            throwErrorMessage(lineData.size(), 5, "You must specify all the fields for bounds!");
            bounds = parseBounds(lineData);
        }

        else if (lineData[0] == "res")
        {
            throwErrorMessage(lineData.size(), 3, "You must specify all the fields for res!");
            res = parseRes(lineData);
        }

        else if (lineData[0] == "sphere")
        {
            throwErrorMessage(lineData.size(), 8, "You must specify all the fields for sphere!");
            parseSphere(lineData);
        }

        else if (lineData[0] == "model")
        {
            throwErrorMessage(lineData.size(), 11, "You must specify all the fields for model!");
            parseModel(lineData);
        }

        else if (lineData[0] == "#" || lineData[0][0] == '#')
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

void DReader::throwErrorMessage(int size, int requiredSize, const string &message)
{
    if(size < requiredSize)
    {
        throw invalid_argument(message);
    }
}

string DReader::findDriverName(const string &file)
{
    unsigned long pIndex = driverFile.find_last_of('.');
    unsigned long sIndex = driverFile.find_last_of('/');
    sIndex += 1;
    return driverFile.substr(sIndex, pIndex - sIndex);
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

void DReader::parseSphere(const vector<string> &lineData)
{
    Eigen::Vector3d position(stod(lineData[1]), stod(lineData[2]), stod(lineData[3]));
    double radius = stod(lineData[4]);
    Eigen::Vector3d albedo(stod(lineData[5]), stod(lineData[6]), stod(lineData[7]));

    int otherProperty = 0;

    for(size_t i = 8; i < lineData.size(); i++)
    {
        if(lineData[i] == "light")
        {
            otherProperty = 1;
        }

        else if(lineData[i] == "mirror")
        {
            otherProperty = 2;
        }

        else if(lineData[i] == "glass")
        {
            otherProperty = 3;
        }
    }

    items.emplace_back(new Sphere(position, radius, Material("A Sphere Material", albedo, otherProperty)));
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

    double smoothingAngle = stod(lineData[9]);
    string modelPath = lineData[10];

    Remap map(rotationVector, theta, scalar, translation, smoothingAngle, modelPath);
    items.emplace_back(new Object(map));
}

ostream &operator<<(ostream &out, const DReader &driver)
{
    Eigen::IOFormat ArrayFormat(Eigen::StreamPrecision, 0, "", ", ", "", "", "[", "]");

    cout << '\n' << "Scene Setup" << '\n' << "-----------" << '\n';

    out << "Driver File: " << driver.driverFile << '\n';
    out << "Driver FileName: " << driver.driverName << '\n';
    out << "Driver Camera: " << driver.camera << '\n';
    out << "Driver Ambient Light: " << driver.ambientLight.format(ArrayFormat) << '\n';

    cout << '\n' << "Scene Items" << '\n' << "-----------" << '\n';

    for(const auto &item : driver.items)
    {
        if(dynamic_cast<Sphere *>(&*item))
        {
            out << "Sphere: \n" << *dynamic_cast<Sphere *>(&*item) << '\n';
        }

        else if(dynamic_cast<Object *>(&*item))
        {
            out << "Object:" << *dynamic_cast<Object *>(&*item) << '\n';
        }
    }

    return out;
}
