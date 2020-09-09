//
// Created by Max Rosoff on 9/7/2019.
//

#include "DReader.h"

using namespace std;

DReader::DReader(const string &file)

: driverFile(file), driverName(findDriverName(file))

{
    readDriver(file);
}

void DReader::readDriver(const string &file)
{
    ifstream driverReader(file);

    if (!driverReader)
    {
        string err = strerror(errno);
        throw invalid_argument("Failure to open Driver File - " + file + ": " + err);
    }

    Vector<3> eye;
    Vector<3> look;
    Vector<3> up;
    double focLen = 0;
    Vector<4> bounds;
    Vector<2> res;

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

Vector<3> DReader::parseEye(const vector<string> &lineData) const
{
    Vector<3> eyeVector;

    eyeVector[0] = stod(lineData[1]);
    eyeVector[1] = stod(lineData[2]);
    eyeVector[2] = stod(lineData[3]);

    return eyeVector;
}

Vector<3> DReader::parseLook(const vector<string> &lineData) const
{
    Vector<3> lookVector;

    lookVector[0] = stod(lineData[1]);
    lookVector[1] = stod(lineData[2]);
    lookVector[2] = stod(lineData[3]);

    return lookVector;
}

Vector<3> DReader::parseUp(const vector<string> &lineData) const
{
    Vector<3> upVector;

    upVector[0] = stod(lineData[1]);
    upVector[1] = stod(lineData[2]);
    upVector[2] = stod(lineData[3]);

    return upVector;
}

double DReader::parseD(const vector<string> &lineData) const
{
    return stod(lineData[1]);
}

Vector<4> DReader::parseBounds(const vector<string> &lineData) const
{
    Vector<4> bounds;

    bounds[0] = stod(lineData[1]);
    bounds[1] = stod(lineData[2]);
    bounds[2] = stod(lineData[3]);
    bounds[3] = stod(lineData[4]);

    return bounds;
}

Vector<2> DReader::parseRes(const vector<string> &lineData) const
{
    Vector<2> resolution;

    resolution[0] = stod(lineData[1]);
    resolution[1] = stod(lineData[2]);

    return resolution;
}

void DReader::parseSphere(const vector<string> &lineData)
{
    Vector<3> position;
    Vector<3> albedo;

    position[0] = stod(lineData[1]);
    position[1] = stod(lineData[2]);
    position[2] = stod(lineData[3]);

    double radius = stod(lineData[4]);

    albedo[0] = stod(lineData[5]);
    albedo[1] = stod(lineData[6]);
    albedo[2] = stod(lineData[7]);

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

    items.emplace_back(new Sphere(position, radius, Material(0, albedo, otherProperty)));
}

void DReader::parseModel(const vector<string> &lineData)
{
    Vector<3> rotationVector;

    rotationVector[0] = stod(lineData[1]);
    rotationVector[1] = stod(lineData[2]);
    rotationVector[2] = stod(lineData[3]);

    double theta = stod(lineData[4]);

    Matrix<4, 4> scalar;

    Vector<4> scalarLineOne;
    Vector<4> scalarLineTwo;
    Vector<4> scalarLineThree;
    Vector<4> scalarLineFour;

    scalarLineOne[0] = stod(lineData[5]);
    scalarLineOne[1] = 0;
    scalarLineOne[2] = 0;
    scalarLineOne[3] = 0;

    scalarLineTwo[0] = 0;
    scalarLineTwo[1] = stod(lineData[5]);
    scalarLineTwo[2] = 0;
    scalarLineTwo[3] = 0;

    scalarLineThree[0] = 0;
    scalarLineThree[1] = 0;
    scalarLineThree[2] = stod(lineData[5]);
    scalarLineThree[3] = 0;

    scalarLineFour[0] = 0;
    scalarLineFour[1] = 0;
    scalarLineFour[2] = 0;
    scalarLineFour[3] = 1;

    scalar[0] = scalarLineOne;
    scalar[1] = scalarLineTwo;
    scalar[2] = scalarLineThree;
    scalar[3] = scalarLineFour;

    Matrix<4, 4> translation;

    Vector<4> translationLineOne;
    Vector<4> translationLineTwo;
    Vector<4> translationLineThree;
    Vector<4> translationLineFour;

    translationLineOne[0] = 1;
    translationLineOne[1] = 0;
    translationLineOne[2] = 0;
    translationLineOne[3] = stod(lineData[6]);

    translationLineTwo[0] = 0;
    translationLineTwo[1] = 1;
    translationLineTwo[2] = 0;
    translationLineTwo[3] = stod(lineData[7]);

    translationLineThree[0] = 0;
    translationLineThree[1] = 0;
    translationLineThree[2] = 1;
    translationLineThree[3] = stod(lineData[8]);

    translationLineFour[0] = 0;
    translationLineFour[1] = 0;
    translationLineFour[2] = 0;
    translationLineFour[3] = 1;

    translation[0] = translationLineOne;
    translation[1] = translationLineTwo;
    translation[2] = translationLineThree;
    translation[3] = translationLineFour;

    double smoothingAngle = stod(lineData[9]);
    string modelPath = lineData[10];

    Remap map(rotationVector, theta, scalar, translation, smoothingAngle, modelPath);
    items.emplace_back(new Object(map));
}
