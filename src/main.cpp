#include "obj_mesh.h"
#include "subdivision.h"

#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include <igl/triangulated_grid.h>

#include <Eigen/Dense>
#include <glm/glm.hpp>
#include "obj_mesh.h"

Doosabin2Subdivision doosabin_;

std::vector<std::vector<int>> transform_faces(obj_mesh mesh){
    std::vector<std::vector<int>> F;
    for (auto f : mesh.faces){
        std::vector<int> face;
        for (auto vi : f) face.push_back(vi.v_idx);
        F.push_back(face);
    }
    return F;
}

void to_obj_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, obj_mesh &_obj){
    _obj.positions.clear();
    _obj.normals.clear();
    _obj.texcoords.clear();
    _obj.faces.clear();
    
    for (int vi=0; vi<V.rows(); vi++) {
        _obj.positions.push_back(glm::vec3(V(vi,0),V(vi,1),V(vi,2)));
    }
    for (int fi=0; fi<F.rows(); fi++) {
        std::vector<vertex_index> face;
        for (int vi=0;vi<F.cols();vi++){
            vertex_index idx;
            idx.v_idx = (unsigned int) F(fi,vi);
            face.push_back(idx);
        }
        _obj.faces.push_back(face);
    }
}

void myCallback() {
    ImGui::PushItemWidth(100); 

    ImGui::PopItemWidth();

    if (ImGui::Button("Doo-Sabin Subdivision Step")) {
        obj_mesh mesh = doosabin_.execute(1);
        std::vector<std::vector<int>> F = transform_faces(mesh);
        polyscope::registerSurfaceMesh("Do-Sabin Subdiv Surface", mesh.positions, F);
    }

}

int main(int argc, char *argv[])
{

    /*
    obj_mesh mesh;
    std::string mesh_path = "../models/tetrahedron.obj";
    if (argc > 1) mesh_path = argv[1];
	loadObj(mesh_path, mesh);
	doosabin_.loadMesh(mesh);
    std::vector<std::vector<int>> F = transform_faces(mesh);
    */

    int n = 4;
    int m = 5;
    Eigen::MatrixXd CP(n*m,3);
    Eigen::MatrixXi CPquads((n-1)*(m-1),4);
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            CP.row(i*m+j) = Eigen::RowVector3d(-0.5 + (1./n)*i, -0.5 + (1./m)*j, 0.1*sin(2*i*M_PI/n)*sin(2*j*M_PI/m) );

            if ((i<n-1) && (j<m-1)){
                auto idx = [n,m](int i, int j) {return i*m+j;};
                CPquads.row(i*(m-1)+j) = Eigen::RowVector4i(idx(i,j),idx(i+1,j),idx(i+1,j+1),idx(i,j+1));
            }
        }
    }
    
    // Eigen::MatrixXd Vb;
    // Eigen::MatrixXi Fb;
    // igl::triangulated_grid(10, 10, Vb, Fb);

    obj_mesh mesh;
    to_obj_mesh(CP, CPquads, mesh);
	doosabin_.loadMesh(mesh);
    std::vector<std::vector<int>> F = transform_faces(mesh);

    // -----------------------------
    // --------- POLYSCOPE ---------
    // -----------------------------

    polyscope::init();
    polyscope::view::upDir = polyscope::UpDir::ZUp;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
    polyscope::options::shadowBlurIters = 6;

    auto pc = polyscope::registerSurfaceMesh("Do-Sabin Subdiv Surface", mesh.positions, F);

    polyscope::registerSurfaceMesh("Control Mesh", CP, CPquads);

    polyscope::state::userCallback = myCallback;

    polyscope::show();

	return 0;
}
