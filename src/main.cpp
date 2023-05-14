#include "obj_mesh.h"
#include "subdivision.h"

#include "polyscope/surface_mesh.h"

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

void myCallback() {
    ImGui::PushItemWidth(100); 

    ImGui::PopItemWidth();

    if (ImGui::Button("Doo-Sabin Subdivision Step")) {
        // executes when button is pressed
        // mySubroutine();
        obj_mesh mesh = doosabin_.execute(1);
        std::vector<std::vector<int>> F = transform_faces(mesh);
        polyscope::registerSurfaceMesh("Mesh", mesh.positions, F);
    }

}

int main(int argc, char *argv[])
{

    obj_mesh mesh;
    std::string mesh_path = "../models/tetrahedron.obj";
    if (argc > 1) mesh_path = argv[1];
	loadObj(mesh_path, mesh);

	doosabin_.loadMesh(mesh);

    // -----------------------------
    // --------- POLYSCOPE ---------
    // -----------------------------

    std::vector<std::vector<int>> F = transform_faces(mesh);

    polyscope::init();
    polyscope::view::upDir = polyscope::UpDir::ZUp;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
    polyscope::options::shadowBlurIters = 6;

    auto pc = polyscope::registerSurfaceMesh("Mesh", mesh.positions, F);

    polyscope::state::userCallback = myCallback;

    polyscope::show();

	return 0;
}
