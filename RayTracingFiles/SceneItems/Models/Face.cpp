//
// Created by Max Rosoff on 9/7/2019.
//

#include "Face.h"

using namespace std;

Face::Face(const vector<int> &vertexIndexList, const int materialIndex) :

vertexIndexList(vertexIndexList), materialIndex(materialIndex)

{}

ostream &operator<<(ostream &out, const Face &face)
{
    for(const int vertex : face.vertexIndexList)
    {
        out << vertex << ' ';
    }
    
    out << '\n';

    return out;
}
