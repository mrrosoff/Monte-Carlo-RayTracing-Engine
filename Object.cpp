//
// Created by Max Rosoff on 9/7/2019.
//

#include "Object.h"

using namespace std;

Object::Object(const Remap &map) :

objPath(map.objPath)

{
    readObject(map);
}

void Object::readObject(const Remap &map) {

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
        vector<string> line;

        stringstream tokenizer(objLine);
        string token;

        while (getline(tokenizer, token, ' '))
        {
            line.push_back(token);
        }

        if (line[0] == "#")
        {
            continue;
        }

        else if (line[0] == "mtllib")
        {
            ifstream matReader(line[1]);

	          if (!matReader)
    	      {
        	     string err = strerror(errno);
        	     throw invalid_argument("Failure to open Material File - " + line[1] + ": " + err);
    	      }

            string matLine;
            string name;
            Eigen::Vector3d Ka;
            Eigen::Vector3d Kd;
            Eigen::Vector3d Ks;
            Eigen::Vector3d Kr;
            double Ns = 0;
            double Ni = 0;
            double d = 0;
            int illum = 0;

            while(getline(matReader, matLine))
            {
                if(matLine.empty())
                {
                    if(illum != 0)
                    {
                        if(illum == 2)
                        {
                            Kr = {0, 0, 0};
                        }

                        else if(illum == 3)
                        {
                            Kr = Ks;
                        }

                        materials.emplace_back(name, Ka, Kd, Ks, Kr, Ns, Ni, d, illum);
                    }

                    continue;
                }

                vector<string> matLineData;
                stringstream matTokenizer(matLine);
                string matToken;

                while (getline(matTokenizer, matToken, ' '))
                {
                    matLineData.push_back(matToken);
                }

                if(matLineData[0] == "#")
                {
                    continue;
                }

                if(matLineData[0] == "newmtl")
                {
                    name = matLineData[1];
                }

                else if(matLineData[0] == "Ka")
                {
                    Ka << stod(matLineData[1]),
                          stod(matLineData[2]),
                          stod(matLineData[3]);
                }

                else if(matLineData[0] == "Kd")
                {
                    Kd << stod(matLineData[1]),
                          stod(matLineData[2]),
                          stod(matLineData[3]);
                }

                else if(matLineData[0] == "Ks")
                {
                    Ks << stod(matLineData[1]),
                          stod(matLineData[2]),
                          stod(matLineData[3]);
                }

                else if(matLineData[0] == "Ns")
                {
                    Ns = stod(matLineData[1]);
                }

                else if(matLineData[0] == "Ni")
                {
                    Ni = stod(matLineData[1]);
                }

                else if(matLineData[0] == "d")
                {
                    d = stod(matLineData[1]);
                }

                else if(matLineData[0] == "illum")
                {
                    illum = stoi(matLineData[1]);
                }
            }
        }

        else if (line[0] == "usemtl")
        {
            currentMatIndex++;
        }

        else if (line[0] == "v")
        {
            Eigen::Vector4d beforeVector(stod(line[1]), stod(line[2]), stod(line[3]), 1);
            auto afterVector = map.transformation * beforeVector;
            vertices.emplace_back(afterVector[0], afterVector[1], afterVector[2]);
        }

        else if (line[0] == "vn")
        {
            continue;
        }

        else if (line[0] == "f")
        {
            vector<vector<int>> face;

            for(size_t i = 1; i < line.size(); i++)
            {
                string faceElement = line[i];
                int positionOfSlashes = faceElement.find("//");

                string leftFace = faceElement.substr(0, positionOfSlashes);
                faceElement.erase(0, positionOfSlashes + 2);
                string rightFace = faceElement.substr(0, faceElement.size());

                vector<int> faceInfo;
                faceInfo.push_back(stoi(leftFace));
                faceInfo.push_back(stoi(rightFace));

                face.push_back(faceInfo);
            }

            faces.push_back(tuple<vector<vector<int>>, int>(face, currentMatIndex));
        }
    }
}

bool Object::intersectionTest(Ray &ray) const
{
    bool foundFace = false;
    const double EPSILON = 1 * pow(10, -5);

    for(const auto &face : faces)
    {
        vector<vector<int>> faceInfo; int matIndex;
        tie (faceInfo, matIndex) = face;

        auto vertexOne = vertices[faceInfo[0][0] - 1];
        auto vertexTwo = vertices[faceInfo[1][0] - 1];
        auto vertexThree = vertices[faceInfo[2][0] - 1];

        Eigen::Vector3d b = vertexOne - ray.point;
        auto columnOne = vertexOne - vertexTwo;
        auto columnTwo = vertexOne - vertexThree;

        Eigen::Matrix3d a;
        a << columnOne[0], columnTwo[0], ray.direction[0],
             columnOne[1], columnTwo[1], ray.direction[1],
             columnOne[2], columnTwo[2], ray.direction[2];

        Eigen::Vector3d x = a.inverse() * b;

        if(x[0] >= EPSILON && x[1] >= EPSILON && x[0] + x[1] <= 1 - EPSILON && x[2] > EPSILON)
        {
            if(x[2] < ray.closestIntersectionDistance)
            {
                ray.material = materials[matIndex];
                ray.closestIntersectionDistance = x[2];
                ray.closestIntersectionPoint = ray.point + x[2] * ray.direction;
                ray.surfaceNormal = (columnOne.cross(columnTwo)).normalized();

                if(ray.direction.dot(ray.surfaceNormal) > 0)
                {
                    ray.surfaceNormal = ray.surfaceNormal * -1;
                }

                foundFace = true;
            }
        }
    }

    return foundFace;
}

ostream &operator<<(ostream &out, const Object &object)
{
    out << '\n';

    for(const auto &material : object.materials)
    {
        out << material << '\n';
    }

    return out;
}
