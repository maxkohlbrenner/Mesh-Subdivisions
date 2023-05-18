#include "obj_mesh.h"
#include "subdivision.h"

#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include <igl/triangulated_grid.h>
#include <igl/readOBJ.h>
#include <igl/gaussian_curvature.h>
#include <igl/octree.h>
#include <igl/knn.h>

#include <Eigen/Dense>
#include <glm/glm.hpp>
#include "obj_mesh.h"

// Doosabin2Subdivision subdivision;
obj_mesh mesh_init, mesh_cur;
Subdivision *subdivision_ = new Doosabin2Subdivision();

Eigen::MatrixXd Vr;
Eigen::MatrixXi Fr;

// octree: (stores reference mesh positions for fast distance computation)
struct octree{
    std::vector<std::vector<int>> point_indices;
    Eigen::MatrixXi CH;
    Eigen::MatrixXd CN;
    Eigen::VectorXd   W;
} tree;

bool show_error = false;

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

Eigen::VectorXd vertex_errors(const Eigen::MatrixXd &Vr, const obj_mesh &mesh) {
    Eigen::VectorXd errors = Eigen::VectorXd::Zero(mesh.positions.size());
    Eigen::MatrixXd P(mesh.positions.size(),3);
    for (int vi=0; vi<mesh.positions.size(); vi++) {
        P.row(vi) = Eigen::Vector3d(mesh.positions[vi][0],mesh.positions[vi][1],mesh.positions[vi][2]);
    }
    Eigen::MatrixXi I;
    igl::knn(P,Vr,1,tree.point_indices, tree.CH, tree.CN, tree.W,I);
    errors = (P - Vr(I.col(0),Eigen::all)).rowwise().norm();

    return errors;
}

static void HelpMarker(const char* desc)
{
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void myCallback() {
    ImGui::PushItemWidth(100); 

    ImGui::PopItemWidth();

    const char* items[] = { "Doo-Sabin", "Catmull-Clarke"};
    static int item_current = 0;

    bool update_mesh  = false;
    bool recalc_error = false;

    if (ImGui::Combo("Subdivision Method", &item_current, items, IM_ARRAYSIZE(items))) {
        if (item_current == 0) {
            subdivision_ = new Doosabin2Subdivision();
        } else if (item_current == 1) {
            subdivision_ = new CatmullSubdivision();
        } else {
            std::cout << "Error: subdivison xethod out of range" << std::endl;
        }

        mesh_cur = mesh_init;
        subdivision_->loadMesh(mesh_cur);
        update_mesh = true;
        if (show_error) {
            recalc_error = true;
        }
    }
    // ImGui::SameLine(); HelpMarker("Refer to the \"Combo\" section below for an explanation of the full BeginCombo/EndCombo API, and demonstration of various flags.\n");


    if (ImGui::Button("Subdivide")) {
        mesh_cur = subdivision_->execute(1);
        update_mesh = true;
    }


    if (ImGui::Checkbox("Show Error" , &show_error)){
        std::cout << "Show error: " << show_error << std::endl;
        if (show_error) {
            recalc_error = true;
        }
    }

    if (update_mesh) {
        std::vector<std::vector<int>> F = transform_faces(mesh_cur);
        auto sds = polyscope::registerSurfaceMesh("Subdivision Surface", mesh_cur.positions, F);

        if (show_error) {
            recalc_error = true;
        }
    }
    if (recalc_error) {
        Eigen::VectorXd err = vertex_errors(Vr, mesh_cur);
        polyscope::getSurfaceMesh("Subdivision Surface")->addVertexScalarQuantity("error", err)->setEnabled(true);
    }

    if (ImGui::Button("Show Gaussian Curvature")) {

        Eigen::MatrixXd Vt;
        Eigen::MatrixXi Ft;
        std::vector<std::vector<int>> F = transform_faces(mesh_cur);
        int nv = mesh_cur.positions.size();
        Vt.resize(nv,3);
        for (int vi=0;vi<nv;vi++){
           for (int d=0; d<3; d++) Vt(vi,d) = mesh_cur.positions[vi][d];
        }
        std::vector<Eigen::Vector3i> trifaces;
        for (int fi=0; fi<F.size(); fi++){
            for (int vj = 1; vj<F[fi].size()-1; vj++){
                trifaces.push_back(Eigen::Vector3i(F[fi][0],F[fi][vj],F[fi][vj+1]));
            }
        }
        Ft.resize(trifaces.size(),3);
        for (int fi=0; fi<trifaces.size(); fi++){
            Ft.row(fi) = trifaces[fi];
        }

        Eigen::VectorXd K;
        igl::gaussian_curvature(Vt,Ft,K);
        polyscope::getSurfaceMesh("Subdivision Surface")->addVertexScalarQuantity("Gaussian Curvature", K)->setEnabled(true);

        Eigen::VectorXd Kr;
        igl::gaussian_curvature(Vr,Fr,Kr);
        polyscope::getSurfaceMesh("Reference Surface")->addVertexScalarQuantity("Gaussian Curvature", Kr)->setEnabled(true);

    }
}


int main(int argc, char *argv[])
{

    // load meshes
    std::string mesh_path           = "../../g1-quad-beziers/spot_tobi/out.obj";
    std::string reference_mesh_path = "../../g1-quad-beziers/spot_tobi/spot_final_tesselated.obj";  // spot_tobi_cc5.obj
    std::string our_cp_mesh_path   =  "../../g1-quad-beziers/spot_tobi/out_ours.obj"; 

	loadObj(mesh_path, mesh_init);
    igl::readOBJ(reference_mesh_path,Vr,Fr);
    igl::octree(Vr, tree.point_indices, tree.CH, tree.CN, tree.W);

    std::cout << "Vr.shape: " << Vr.rows() << ", " << Vr.cols() << std::endl;

    Eigen::MatrixXd Vours;
    Eigen::MatrixXi Fours;
    igl::readOBJ(our_cp_mesh_path,Vours,Fours);

    mesh_cur = mesh_init;
	subdivision_->loadMesh(mesh_cur);
    std::vector<std::vector<int>> F = transform_faces(mesh_cur);
    std::vector<std::vector<int>> F_init = transform_faces(mesh_init);

    polyscope::init();
    polyscope::view::upDir = polyscope::UpDir::ZUp;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
    polyscope::options::shadowBlurIters = 6;

    auto pc = polyscope::registerSurfaceMesh("Subdivision Surface", mesh_cur.positions, F);
    polyscope::registerSurfaceMesh("Reference Surface", Vr, Fr);

    polyscope::registerPointCloud("K-Surf Control Points", Vours);

    polyscope::state::userCallback = myCallback;

    polyscope::show();

	return 0;
}
