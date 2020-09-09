//
// Created by Max Rosoff on 9/7/2019.
//

#include "Object.h"

using namespace std;

Object::Object(const Remap &map) :

objPath(map.path), smoothingAngle(map.smoothingAngle)

{
    readObject(map);
}

void Object::readObject(const Remap &map)
{
    ifstream objReader(objPath);

    if (!objReader)
    {
        string err = strerror(errno);
        throw invalid_argument("Failure to open Object File - " + objPath + ": " + err);
    }

    string objLine;
    int currentMatIndex = -1;

    while(getline(objReader, objLine))
    {
        objLine.erase(remove(objLine.begin(), objLine.end(), '\r'), objLine.end());

        if(objLine.empty())
        {
            continue;
        }

        vector<string> line;

        stringstream tokenizer(objLine);
        string token;

        while (getline(tokenizer, token, ' '))
        {
            if(!token.empty())
            {
                line.emplace_back(token);
            }
        }

        if (line[0] == "#")
        {
            continue;
        }

        else if (line[0] == "mtllib")
        {
            readMaterialFile(line[1]);
        }

        else if (line[0] == "usemtl")
        {
            for(size_t i = 0; i < materials.size(); i++)
            {
                if(materials[i].name == line[1])
                {
                    currentMatIndex = i;
                }
            }
        }

        else if (line[0] == "v")
        {
            Vector<4> beforeVector;

            beforeVector[0] = stod(line[1]);
            beforeVector[1] = stod(line[2]);
            beforeVector[2] = stod(line[3]);
            beforeVector[3] = 1;

            auto afterVector = map.transformation * beforeVector;

            Vector<3> removeLastElement;

            removeLastElement[0] = afterVector[0];
            removeLastElement[1] = afterVector[1];
            removeLastElement[2] = afterVector[2];

            vertices.emplace_back(removeLastElement);
        }

        else if (line[0] == "vn")
        {
            continue;
        }

        else if (line[0] == "f")
        {
            readObjectFaceLine(line, currentMatIndex);
        }
    }

    calculateNormals();
}

void Object::readObjectFaceLine(const vector<string> &line, const int currentMatIndex)
{
    vector<int> face;

    for(size_t i = 1; i < line.size(); i++)
    {
        stringstream faceTokenizer(line[i]);
        string faceToken;

        getline(faceTokenizer, faceToken, '/');

        if(!faceToken.empty())
        {
            face.push_back(stoi(faceToken));
        }
    }

    for(int vertexIndex : face)
    {
        vertices[vertexIndex - 1].adjacentFaces.emplace_back(faces.size());
    }

    faces.emplace_back(face, currentMatIndex);
}

void Object::readMaterialFile(const string &filePath) {

    ifstream matReader(filePath);

    if (!matReader)
    {
        string err = strerror(errno);
        throw invalid_argument("Failure to open Material File - " + filePath + ": " + err);
    }

    string matLine;
    string name;
    Vector<3> albedo;
    int otherProperty = 0;

    while(getline(matReader, matLine))
    {
        matLine.erase(remove(matLine.begin(), matLine.end(), '\r'), matLine.end());

        if(matLine.empty())
        {
            if(!name.empty())
            {
                materials.emplace_back(name, albedo, otherProperty);
                name.clear();
            }

            continue;
        }

        vector<string> matLineData;
        stringstream matTokenizer(matLine);
        string matToken;

        while(getline(matTokenizer, matToken, ' '))
        {
            matLineData.push_back(matToken);
        }

        if(matLineData[0] == "#")
        {
            continue;
        }

        if(matLineData[0] == "newmtl")
        {
            for(const auto &material : materials)
            {
                if(material.name == matLineData[1])
                {
                    throw invalid_argument("It is Illegal To Have Mulitple Materials Of The Same Name!");
                }
            }

            name = matLineData[1];
        }

        else if(matLineData[0] == "albedo")
        {
            albedo[0] = stod(matLineData[1]);
            albedo[1] = stod(matLineData[2]);
            albedo[2] = stod(matLineData[3]);
        }

        else if(matLineData[0] == "light")
        {
            otherProperty = 1;
        }

        else if(matLineData[0] == "mirror")
        {
            otherProperty = 2;
        }

        else if(matLineData[0] == "glass")
        {
            otherProperty = 3;
        }
    }

    if(matLine.empty())
    {
        if(!name.empty())
        {
            materials.emplace_back(name, albedo, otherProperty);
        }
    }
}

void Object::calculateNormals()
{
    for(auto &face: faces)
    {
        if(!face.calcColumns)
        {
            face.columnOne = vertices[face.vertexIndexList[0] - 1].vertex - vertices[face.vertexIndexList[1] - 1].vertex;
            face.columnTwo = vertices[face.vertexIndexList[0] - 1].vertex - vertices[face.vertexIndexList[2] - 1].vertex;
            face.calcColumns = true;
        }

        auto F1Normal = (face.columnOne.cross(face.columnTwo)).normalize();
        
        for(const auto vertex : face.vertexIndexList)
        {
            Vector<3> sum;

            sum[0] = 0;
            sum[1] = 0;
            sum[2] = 0;

            for(const auto faceIndex : vertices[vertex - 1].adjacentFaces)
            {
                auto &otherFace = faces[faceIndex];

                if(!otherFace.calcColumns)
                {
                    otherFace.columnOne = vertices[otherFace.vertexIndexList[0] - 1].vertex - vertices[otherFace.vertexIndexList[1] - 1].vertex;
                    otherFace.columnTwo = vertices[otherFace.vertexIndexList[0] - 1].vertex - vertices[otherFace.vertexIndexList[2] - 1].vertex;
                    otherFace.calcColumns = true;
                }

                auto F2Normal = (otherFace.columnOne.cross(otherFace.columnTwo)).normalize();

                if(abs(acos(min(max(F1Normal.dot(F2Normal), -1.0), 1.0)) * (180 / M_PI)) <= smoothingAngle)
                {
                    sum += F2Normal;
                }
            }

            face.normals.emplace_back(sum.normalize());
        }
    }
}

__device__ Ray Object::makeExitRefrationRay(const Ray &invRay, double originalIndex, double newIndex) const
{
    Vector<3> refractionDirection = doSnellsLaw(invRay.direction, invRay.surfaceNormal, originalIndex, newIndex);
    Ray innerRefractionRay(invRay.closestIntersectionPoint, refractionDirection);
    intersectionTest(innerRefractionRay);
    return Ray(innerRefractionRay.closestIntersectionPoint, doSnellsLaw(refractionDirection * -1, innerRefractionRay.surfaceNormal, newIndex, originalIndex));
}

__device__ bool Object::intersectionTest(Ray &ray) const
{
    bool foundFace = false;
    const double EPSILON = 1 * pow(10, -5);

    for(const auto &face : faces)
    {
        const auto b = vertices[face.vertexIndexList[0] - 1].vertex - ray.point;

        Matrix<3, 3> a;

        Vector<3> aLineOne;
        Vector<3> aLineTwo;
        Vector<3> aLineThree;

        aLineOne[0] = face.columnOne[0];
        aLineOne[1] = face.columnTwo[0];
        aLineOne[2] = ray.direction[0];

        aLineTwo[0] = face.columnOne[1];
        aLineTwo[1] = face.columnTwo[1];
        aLineTwo[2] = ray.direction[1];

        aLineThree[0] = face.columnOne[2];
        aLineThree[1] = face.columnTwo[2];
        aLineThree[2] = ray.direction[2];

        a[0] = aLineOne;
        a[1] = aLineTwo;
        a[2] = aLineThree;

        const auto x = a.inverse() * b;

        if(x[0] >= EPSILON && x[1] >= EPSILON && x[0] + x[1] <= 1 - EPSILON && x[2] > EPSILON)
        {
            if(x[2] < ray.closestIntersectionDistance)
            {
                ray.closestIntersectionDistance = x[2];

                ray.hit = this;
                ray.material = materials[face.materialIndex];
                ray.surfaceNormal = (face.normals[0] * (1.0 - x[0] - x[1]) + face.normals[1] * x[0] + face.normals[2] * x[1]).normalize();
                ray.closestIntersectionPoint = ray.point + ray.direction * x[2];

                if(ray.direction.dot(ray.surfaceNormal) > 0)
                {
                    ray.surfaceNormal *= -1;
                }

                foundFace = true;
            }
        }
    }
    
    return foundFace;
}
