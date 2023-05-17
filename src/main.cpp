#include "obj_mesh.h"
#include "subdivision.h"

#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include <igl/triangulated_grid.h>
#include <igl/readOBJ.h>

#include <Eigen/Dense>
#include <glm/glm.hpp>
#include "obj_mesh.h"

// Doosabin2Subdivision subdivision;
obj_mesh mesh_init, mesh_cur;
Subdivision *subdivision_ = new Doosabin2Subdivision();

Eigen::MatrixXd Vr;
Eigen::MatrixXi Fr;

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

    // Eigen::MatrixXd ref_test(1,3);
    // ref_test.row(0) = Eigen::RowVector3d(mesh.positions[0][0],mesh.positions[0][1],mesh.positions[0][2]);
    // for ( int i=0; i<5; i++) ref_test.row(i) = Eigen::RowVector3d(2.,0.,0.);
    for (int i=0; i<mesh.positions.size(); i++) {
        Eigen::Vector3d p(mesh.positions[i][0],mesh.positions[i][1],mesh.positions[i][2]);
        errors[i] = (Vr.array().rowwise() - p.transpose().array()).rowwise().norm().minCoeff();
        // errors[i] = (ref_test.array().rowwise() - p.transpose().array()).rowwise().norm().minCoeff();
    }
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
        if (show_error) {
            recalc_error = true;
        }
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
    }
    if (recalc_error) {
        Eigen::VectorXd err = vertex_errors(Vr, mesh_cur);
        polyscope::getSurfaceMesh("Subdivision Surface")->addVertexScalarQuantity("error", err)->setEnabled(true);
    }
}


int main(int argc, char *argv[])
{

    // load meshes
    std::string mesh_path           = "../../g1-quad-beziers/spot_tobi/out.obj";
    std::string reference_mesh_path = "../../g1-quad-beziers/spot_tobi/spot_tobi_cc5.obj"; 
    std::string our_cp_mesh_path   =  "../../g1-quad-beziers/spot_tobi/out_ours.obj"; 

	loadObj(mesh_path, mesh_init);
    igl::readOBJ(reference_mesh_path,Vr,Fr);

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
