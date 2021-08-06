#include <fstream>
#include <string>
#include <vector_types.h>
#include <vector_functions.h>

#include "ply_io.h"
#include "logging.h"
#include "ply_utils.h"
// CHECK DONE headers follow rules in NOTES

bool checkIfPlyFile(const std::string& filepath)
{
  std::ifstream file(filepath, std::ios::in);
  bool headerIsPly = false;
  if (file.is_open() && file.good()) {

    std::string line;
    while (std::getline(file, line)) {
      if (!line.empty()) {

  	headerIsPly = line.compare("ply") == 0;
  	if (headerIsPly) {
  	  break;
  	}
      }
    }
    file.close();
  }
  return headerIsPly;
}


PlyFile* loadPlyFile(const char* filepath)
{
  PlyFile *ply = NULL;
  FILE *fp = fopen(filepath, "r");
  if (fp != NULL) {
    ply = read_ply(fp);
    if (ply != NULL) {
      LOG_TRIVIAL(debug) << "PLY file " << filepath << " read successfully.";
    }
    else {
      LOG_TRIVIAL(error) << "Could not read PLY data from "
			 << filepath << ".";
    }
  }
  else {
    LOG_TRIVIAL(error) << "Could not open " << filepath << ".";
  }
  return ply;
}

typedef struct {
  unsigned char nIndices;
  int *indices;
  unsigned char nTexCoords;
  float *texCoords;
} triangle_t;

typedef struct {
  float x;
  float y;
  float z;
  unsigned char r;
  unsigned char g;
  unsigned char b;
} vertex_t;

bool propInList(const char* searchedName, PlyProperty **elemProps, int nprops)
{
  bool propFound = false;
  for (int propIdx = 0; propIdx < nprops; propIdx++) {
    std::string propName(elemProps[propIdx]->name);
    if (propName.compare(searchedName) == 0) {
      propFound = true;
      break;
    }
  }
  return propFound;
}


void getProperties(PlyFile* ply, char* elemNameCstr,
		   PlyProperty* usedProps,
		   int numUsedProps, PlyProperty** elemProps,
		   int nprops, std::vector<int>& propsPresent)
{
  LOG_TRIVIAL(debug) << "PLY: get_properties: element name: " << elemNameCstr;
  for (int usedPropIdx = 0; usedPropIdx < numUsedProps; usedPropIdx++) {
    char *usedPropName = usedProps[usedPropIdx].name;
    LOG_TRIVIAL(debug) << "PLY: property name: " << usedPropName;
    if (propInList(usedPropName, elemProps, nprops)) {
      ply_get_property(ply, elemNameCstr, &usedProps[usedPropIdx]);
      propsPresent[usedPropIdx] = 1;
      LOG_TRIVIAL(debug) << "PLY: \t" << usedPropName << " is present.";
    }
  }
}

void loadPlyObject(const char* filepath,
		   std::vector<float3>& vertices,
		   std::vector<float3>& vertexColors,
		   std::vector<float3>& vertexNormals,
		   std::vector<float2>& textureCoords,
		   std::vector<int>& triangleIndices,
		   StatusCode& status)
{
  PlyFile *ply = loadPlyFile(filepath);
  if (ply == NULL) {
    status = StatusCode::FileError;
    return;
  }

  int numElems;
  char **elementNameList = get_element_list_ply(ply, &numElems);
  LOG_TRIVIAL(debug) << "PLY: numElems " << numElems;
  for (int elemIdx = 0; elemIdx < numElems; elemIdx++) {
    LOG_TRIVIAL(debug) << "PLY: elem[" << elemIdx << "]: "
		       << elementNameList[elemIdx];
  }

  //--------------------------------------------------------------------------
  // list of property information for a vertex
  PlyProperty vertProps[] =
    {{"x", Float32, Float32, offsetof(vertex_t,x), 0, 0, 0, 0},
     {"y", Float32, Float32, offsetof(vertex_t,y), 0, 0, 0, 0},
     {"z", Float32, Float32, offsetof(vertex_t,z), 0, 0, 0, 0},
     {"red", Uint8, Uint8, offsetof(vertex_t,r), 0, 0, 0, 0},
     {"green", Uint8, Uint8, offsetof(vertex_t,g), 0, 0, 0, 0},
     {"blue", Uint8, Uint8, offsetof(vertex_t,b), 0, 0, 0, 0},};
  int numVertProps = sizeof(vertProps)/sizeof(PlyProperty);
  std::vector<int> vertPropsPresent(numVertProps, 0);

  //--------------------------------------------------------------------------
  // list of property information for a face - triangle
  PlyProperty faceProps[] =
    {{"vertex_indices", Int32, Int32, offsetof(triangle_t, indices),
      1, Uint8, Uint8, offsetof(triangle_t, nIndices)},
     {"texcoord", Float32, Float32, offsetof(triangle_t, texCoords),
      1, Uint8, Uint8, offsetof(triangle_t, nTexCoords)}
    };
  int numFaceProps = sizeof(faceProps)/sizeof(PlyProperty);
  std::vector<int> facePropsPresent(numFaceProps, 0);

  for (int elem = 0; elem < numElems; elem++) {
    std::string elemName = std::string(elementNameList[elem]);
    char *elemnameCstr = elementNameList[elem];
    int nelems, nprops;
    PlyProperty **elemProps = get_element_description_ply(ply, elemnameCstr,
							  &nelems, &nprops);

    LOG_TRIVIAL(debug) << "PLY: elemName " << elemName
		       << ", nelems: " << nelems
		       << ", nprops: " << nprops;

    if (elemName.compare("vertex") == 0) {

      getProperties(ply, elemnameCstr, vertProps,
		    numVertProps, elemProps, nprops,
		    vertPropsPresent);
      bool colorsPresent = vertPropsPresent[3] != 0 &&
	vertPropsPresent[4] != 0 &&
	vertPropsPresent[5] != 0;
      for (int vertexIdx = 0; vertexIdx < nelems; vertexIdx++) {
	vertex_t vertex;
	ply_get_element(ply, (void*)&vertex);
	vertices.push_back(make_float3(vertex.x, vertex.y, vertex.z));
	if (colorsPresent) {
	  vertexColors.push_back(make_float3(vertex.r/255.0f,
					     vertex.g/255.0f,
					     vertex.b/255.0f));
	}
      }
    }
    else if (elemName.compare("face") == 0) {
      getProperties(ply, elemnameCstr, faceProps,
		    numFaceProps, elemProps, nprops,
		    facePropsPresent);
      bool texcoordsPresent = facePropsPresent[1] != 0;
      if (texcoordsPresent) {
      	LOG_TRIVIAL(trace) << "PLY: texture coordinates present.";
      }

      for (int faceIdx = 0; faceIdx < nelems; faceIdx++) {
	triangle_t triangle;
	ply_get_element(ply, (void*)&triangle);
	assert(triangle.nIndices == 3);
	for (int vIdx = 0; vIdx < triangle.nIndices; vIdx++) {
	  triangleIndices.push_back(triangle.indices[vIdx]);
	}
	if (texcoordsPresent) {
	  assert(triangle.nTexCoords == 6);
	  for (int tIdx = 0; tIdx < triangle.nTexCoords/2; tIdx++) {
	    textureCoords.push_back(make_float2(triangle.texCoords[tIdx*2],
						triangle.texCoords[tIdx*2+1]));
	  }
	}
      }
    }
  }
  close_ply(ply);
}
