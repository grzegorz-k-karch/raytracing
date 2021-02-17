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
    fclose(fp);
  }
  else {
    LOG_TRIVIAL(error) << "Could not open " << filepath << ".";
  }
  return ply;
}

void loadPlyObject(const char* filepath,
		   float** vertices, float** vertex_colors,		   
		   float** vertex_normals, int** indices)
{
  PlyFile *ply = loadPlyFile(filepath);

  // PlyProperty vert_props[] = { /* list of property information for a vertex */
  //   {"x", PLY_FLOAT, PLY_FLOAT, 0, 0, 0, 0, 0},
  //   {"y", PLY_FLOAT, PLY_FLOAT, 4, 0, 0, 0, 0},
  //   {"z", PLY_FLOAT, PLY_FLOAT, 8, 0, 0, 0, 0},
  // };
  // /* list of property information for a vertex */
  // PlyProperty face_props[] = {{"vertex_indices", PLY_INT, PLY_INT, offsetof(polygon_t, indices),
  // 			       1, PLY_UCHAR, PLY_UCHAR, offsetof(polygon_t, nindices)}};

  // PlyFile *ply;
  // int nelems;
  // char **elist;
  // int file_type;
  // float version;
  // int nprops;
  // int num_elems;
  // PlyProperty **plist;
  // char *elem_name;
  // bool vertices_aval = false;
  // bool normals_aval = false;
  // bool indices_aval = false;

  // /* open a PLY file for reading */
  // ply = ply_open_for_reading(ply_filename.c_str(), &nelems, &elist, &file_type, &version);

  // std::vector<float> vertices_tmp;
  // std::vector<unsigned> indices_tmp;
  
  // /* go through each kind of element that we learned is in the file */
  // /* and read them */
  // for (int i = 0; i < nelems; i++) {

  //   /* get the description of the first element */
  //   elem_name = elist[i];
  //   plist = ply_get_element_description (ply, elem_name, &num_elems, &nprops);

  //   /* if we're on vertex elements, read them in */
  //   if (equal_strings ("vertex", elem_name)) {

  //     vertices_aval = true;

  //     ply_get_property (ply, elem_name, &vert_props[0]);
  //     ply_get_property (ply, elem_name, &vert_props[1]);
  //     ply_get_property (ply, elem_name, &vert_props[2]);

  //     for (int j = 0; j < num_elems; j++) {

  // 	/* grab and element from the file */
  // 	float vertex[3];
  // 	ply_get_element (ply, (void *) vertex);
  // 	vertices_tmp.push_back(vertex[0]);
  // 	vertices_tmp.push_back(vertex[1]);
  // 	vertices_tmp.push_back(vertex[2]);
  //     }
  //   }
  //   /* if we're on face elements, read them in */
  //   if (equal_strings ("face", elem_name)) {

  //     indices_aval = true;
  //     ply_get_property (ply, elem_name, &face_props[0]);

  //     for (int j = 0; j < num_elems; j++) {

  // 	polygon_t polygon;
  // 	ply_get_element (ply, (void*)&polygon);

  // 	for (int k = 0; k < polygon.nindices; k++) {
  // 	  indices_tmp.push_back(polygon.indices[k]);
  // 	}
  //     }
  //   }
  // }
  // ply_close (ply);

  // if (vertices_aval == false || indices_aval == false) {
  //   std::cerr << "Ply file is corrupt?" << std::endl;
  //   return;
  // }

  // std::map<unsigned,unsigned> merge_map;
  // mergeVertices(vertices_tmp, indices_tmp, 3,
  // 		vertices, indices, merge_map);

  // if (!normals_aval) {
  //   for (int i = 0; i < vertices.size()/4; ++i) {
  //     normals.push_back(1.0f);
  //     normals.push_back(0.0f);
  //     normals.push_back(0.0f);
  //   }
  //   computeNormals(vertices, indices, normals);
  // }
}
