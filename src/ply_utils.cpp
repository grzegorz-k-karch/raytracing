#include <vector_types.h>
#include <vector_functions.h>
#include <string>

#include "ply_io.h"

#include "ply_utils.h"
#include "logging.h"


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
      LOG_TRIVIAL(error) << "Could not read PLY data from " << filepath << ".";
    }
  }
  else {
    LOG_TRIVIAL(error) << "Could not open " << filepath << ".";
  }
  return ply;
}

typedef struct {
  unsigned char nindices;
  int *indices;
} triangle_t;

typedef struct {
  float x;
  float y;
  float z;
  unsigned char r;
  unsigned char g;
  unsigned char b;
} vertex_t;

bool prop_in_list(const char* searched_name, PlyProperty **elem_props, int nprops)
{
  bool prop_found = false;
  for (int prop_idx = 0; prop_idx < nprops; prop_idx++) {
    // char *prop_name = ;
    std::string prop_name(elem_props[prop_idx]->name);
    if (prop_name.compare(searched_name) == 0) {
      prop_found = true;
      break;
    }
  }
  return prop_found;
}


void get_properties(PlyFile* ply, char* elem_name_cstr, PlyProperty* used_props,
		    int num_used_props, PlyProperty** elem_props, int nprops,
		    std::vector<int>& props_present)
{
  for (int used_prop_idx = 0; used_prop_idx < num_used_props; used_prop_idx++) {
    char *used_prop_name = used_props[used_prop_idx].name;
    if (prop_in_list(used_prop_name, elem_props, nprops)) {
      ply_get_property(ply, elem_name_cstr, &used_props[used_prop_idx]);
      props_present[used_prop_idx] = 1;
    }
  }
}

void loadPlyObject(const char* filepath,
		   std::vector<float3>& vertices,
		   std::vector<float3>& vertex_colors,		   
		   std::vector<float3>& vertex_normals,
		   std::vector<int>& triangle_indices)
{
  PlyFile *ply = loadPlyFile(filepath);

  int num_elems;
  char **element_name_list = get_element_list_ply(ply, &num_elems);
  LOG_TRIVIAL(debug) << "PLY: num_elems " << num_elems;

  // list of property information for a vertex
  PlyProperty vert_props[] = {{"x", Float32, Float32, offsetof(vertex_t,x), 0, 0, 0, 0},
  			      {"y", Float32, Float32, offsetof(vertex_t,y), 0, 0, 0, 0},
  			      {"z", Float32, Float32, offsetof(vertex_t,z), 0, 0, 0, 0},
			      {"red", Uint8, Uint8, offsetof(vertex_t,r), 0, 0, 0, 0},
  			      {"green", Uint8, Uint8, offsetof(vertex_t,g), 0, 0, 0, 0},
  			      {"blue", Uint8, Uint8, offsetof(vertex_t,b), 0, 0, 0, 0},};
  int num_vert_props = sizeof(vert_props)/sizeof(PlyProperty);
  std::vector<int> vert_props_present(num_vert_props, 0);

  // list of property information for a vertex
  PlyProperty face_props[] = {{"vertex_indices", Int32, Int32, offsetof(triangle_t, indices),
  			       1, Uint8, Uint8, offsetof(triangle_t, nindices)}};
  int num_face_props = sizeof(face_props)/sizeof(PlyProperty);
  std::vector<int> face_props_present(num_face_props, 0);  
  
  for (int elem = 0; elem < num_elems; elem++) {
    std::string elem_name = std::string(element_name_list[elem]);
    char *elem_name_cstr = element_name_list[elem];
    int nelems, nprops;
    PlyProperty **elem_props = get_element_description_ply(ply, elem_name_cstr, &nelems, &nprops);
  
    LOG_TRIVIAL(debug) << "PLY: elem_name " << elem_name
		       << " nelems: " << nelems
		       << " nprops " << nprops;
    
    if (elem_name.compare("vertex") == 0) {
      
      get_properties(ply, elem_name_cstr, vert_props,
		     num_vert_props, elem_props, nprops,
		     vert_props_present);
      for (int vertex_idx = 0; vertex_idx < nelems; vertex_idx++) {
	vertex_t vertex;
	ply_get_element(ply, (void*)&vertex);
	vertices.push_back(make_float3(vertex.x, vertex.y, vertex.z));
	bool colors_present = vert_props_present[3] != 0 &&
	  vert_props_present[4] != 0 &&
	  vert_props_present[5] != 0;
	if (colors_present) {
	  vertex_colors.push_back(make_float3(vertex.r/255.0f,
					      vertex.g/255.0f,
					      vertex.b/255.0f));
	}
      }
    }
    else if (elem_name.compare("face") == 0) {
      get_properties(ply, elem_name_cstr, face_props,
		     num_face_props, elem_props, nprops,
		     face_props_present);
      for (int face_idx = 0; face_idx < nelems; face_idx++) {
	triangle_t triangle;
	ply_get_element(ply, (void*)&triangle);
	for (int v_idx = 0; v_idx < triangle.nindices; v_idx++) {
	  triangle_indices.push_back(triangle.indices[v_idx]);
	}
      }
    }
  }
  close_ply(ply);
}
