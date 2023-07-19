#include "obj_mesh.h"
#include "subdivision.h"
#include "nanoflann.hpp"

#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include <igl/triangulated_grid.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
// #include <igl/readOBJpoly.h>
#include <igl/gaussian_curvature.h>
// #include <igl/octree.h>
#include <igl/knn.h>

#include <Eigen/Dense>
#include <glm/glm.hpp>
#include "obj_mesh.h"

// Doosabin2Subdivision subdivision;
obj_mesh mesh_init, mesh_cur;
Subdivision *subdivision_ = new Doosabin2Subdivision();

Eigen::MatrixXd Vr;
Eigen::MatrixXi Fr;
double bbx_diagonal;


int step = -1;
std::vector<std::vector<polyscope::Structure*>> structures;

double rotangle = 360.;

/*
// octree: (stores reference mesh positions for fast distance computation)
struct octree{
    std::vector<std::vector<int>> point_indices;
    Eigen::MatrixXi CH;
    Eigen::MatrixXd CN;
    Eigen::VectorXd   W;
} tree;
*/


// nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> flanntree;
using my_kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd>;
my_kd_tree_t *mytree;

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
    /*
    Eigen::MatrixXd P(mesh.positions.size(),3);
    for (int vi=0; vi<mesh.positions.size(); vi++) {
        P.row(vi) = Eigen::Vector3d(mesh.positions[vi][0],mesh.positions[vi][1],mesh.positions[vi][2]);
    }
    */
    for ( int i=0; i<mesh.positions.size(); i++) {

        std::vector<double> query_pt(3);
        Eigen::RowVector3d p;
        for (int d=0;d<3;d++){
            query_pt[d] = mesh.positions[i][d];
            p(d)        = mesh.positions[i][d];
        }

        int num_results = 1;
        std::vector<long unsigned int> ret_indexes(num_results);
        std::vector<double>  out_dists_sqr(num_results);
        nanoflann::KNNResultSet<double> resultSet(1);
        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        mytree->index_->findNeighbors(resultSet, &query_pt[0]);
        
        /*
        std::cout << "Err: " << std::endl;
        std::cout << ret_indexes[0] << std::endl;
        std::cout << sqrt(out_dists_sqr[0]) << std::endl;
        std::cout << (p - Vr.row(ret_indexes[0])).norm() << std::endl;
        */
        errors(i) = sqrt(out_dists_sqr[0]);
    }
    // igl::knn(P,Vr,1,tree.point_indices, tree.CH, tree.CN, tree.W,I);
    // errors = (P - Vr(I.col(0),Eigen::all)).rowwise().norm();

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

void update_struct_visibility(int i){
    for (int si=0; si<structures.size(); si++){
        for (auto sn: structures[si]){
            sn->setEnabled(false);
        }
    }

    for (int si=0; si<structures.size(); si++){
        bool enabled = (si == i);
        if (enabled) for (auto sn: structures[si]) sn->setEnabled(true);
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

        bool byarea = true;

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
                trifaces.push_back(Eigen::Vector3i(F[fi][0],F[fi][vj+1],F[fi][vj]));
            }
        }
        Ft.resize(trifaces.size(),3);
        for (int fi=0; fi<trifaces.size(); fi++){
            Ft.row(fi) = trifaces[fi];
        }


        Eigen::VectorXd K;
        igl::gaussian_curvature(Vt,Ft,K);
        Eigen::SparseMatrix<double> M,Minv;
        if (byarea) {
            igl::massmatrix(Vt,Ft,igl::MASSMATRIX_TYPE_DEFAULT,M);
            igl::invert_diag(M,Minv);
            K = (Minv*K).eval();
        }

        polyscope::getSurfaceMesh("Subdivision Surface")->addVertexScalarQuantity("Gaussian Curvature", K)->setEnabled(true);
        std::cout << "SD: K " << std::endl;
        std::cout << K.minCoeff() << ", " << K.mean() << ", " << K.maxCoeff() << std::endl;

        auto dbg = polyscope::registerSurfaceMesh("SD Tri, debug", Vt, Ft);
        dbg ->addVertexScalarQuantity("Gaussian Curvature", K)->setEnabled(true);

        igl::writeOBJ("mesh_ds.obj", Vt, Ft);

        Eigen::VectorXd Kr;
        igl::gaussian_curvature(Vr,Fr,Kr);
        if (byarea) {
            igl::massmatrix(Vr,Fr,igl::MASSMATRIX_TYPE_DEFAULT,M);
            igl::invert_diag(M,Minv);
            Kr = (Minv*Kr).eval();
        }
        polyscope::getSurfaceMesh("Reference Surface")->addVertexScalarQuantity("Gaussian Curvature", Kr)->setEnabled(true);

        std::cout << "KS: K " << std::endl;
        std::cout << Kr.minCoeff() << ", " << Kr.mean() << ", " << Kr.maxCoeff() << std::endl;

    }


    if (step == -1) {
            if (ImGui::Button("Init animation")) {
                step = 0;
                update_struct_visibility(step);
            }
    } else {
            if (ImGui::Button("Step")) {
                if (step < structures.size()-1){
                    step++;
                } else {
                    step = 0;
                }
                update_struct_visibility(step);
            }
    }

    if (ImGui::Button("Rotate")) {
        rotangle = 0.;
    }

    if (rotangle < 360) {
        float step = 1.;
        rotangle += step;
        auto rs = polyscope::getSurfaceMesh("Reference Surface");
        auto ds = polyscope::getSurfaceMesh("Subdivision Surface");
        glm::mat4x4 T = rs->getTransform();
        rs->setTransform(glm::rotate(T,glm::radians(step) ,glm::vec3(0.,0.,1.)));
        ds->setTransform(glm::rotate(T,glm::radians(step) ,glm::vec3(0.,0.,1.)));
    } else {
        rotangle = 360;
    }


}


void edge_network_from_quadmesh(const Eigen::MatrixXi &F, Eigen::MatrixXi &E){

    E.resize(4*F.rows(),2);
    for (int fi=0; fi<F.rows();fi++) for (int vi=0; vi<4; vi++) E.row(4*fi+vi) = Eigen::RowVector2i(F(fi,vi),F(fi,(vi+1)%4));
}

void patch_edge_network_from_quadmesh(const Eigen::MatrixXi &F, Eigen::MatrixXi &E){

    E.resize(8*F.rows(),2);
    for (int fi=0; fi<F.rows();fi++){
            Eigen::Vector4i f = F.row(fi);

            E.row(8*fi+0)   = Eigen::RowVector2i(f(0),    f(0)+3);
            E.row(8*fi+1)   = Eigen::RowVector2i(f(0)+3,  f(1)  );
            E.row(8*fi+2)   = Eigen::RowVector2i(f(1),    f(1)+1);
            E.row(8*fi+3)   = Eigen::RowVector2i(f(1)+1,  f(2)  );
            E.row(8*fi+4)   = Eigen::RowVector2i(f(2),    f(3)+3);
            E.row(8*fi+5)   = Eigen::RowVector2i(f(3)+3,  f(3)  );
            E.row(8*fi+6)   = Eigen::RowVector2i(f(3),    f(0)+1);
            E.row(8*fi+7)   = Eigen::RowVector2i(f(0)+1,  f(0)  );
    }
}

int main(int argc, char *argv[])
{

    std::string folder = "spot_new";
    std::string prefix = "spot_new";
    // std::string folder = "spot_tobi";
    // std::string prefix = "spot";

    // load meshes
    /*
    std::string mesh_path           = "../../g1-quad-beziers/spot_tobi/out.obj";
    std::string reference_mesh_path = "../../g1-quad-beziers/spot_tobi/spot_tesselated.obj";  // spot_tobi_cc5.obj
    std::string our_cp_mesh_path   =  "../../g1-quad-beziers/spot_tobi/out_ours.obj"; 
    std::string bctrl_quadmesh_path      =  "../../g1-quad-beziers/spot_tobi/spot_bctrl_quadmesh.obj"; 
    std::string spot_quadmesh_path      =  "../../g1-quad-beziers/spot_tobi/spot_bctrl_patches.obj"; 
    */

    std::string mesh_path           = "../../g1-quad-beziers/" + folder + "/" + "out.obj"; 
    std::string reference_mesh_path = "../../g1-quad-beziers/" + folder + "/" + prefix + "_final_tesselated.obj"; 
    std::string our_cp_mesh_path    = "../../g1-quad-beziers/" + folder + "/" + "out_ours.obj"; 
    std::string bctrl_quadmesh_path = "../../g1-quad-beziers/" + folder + "/" + prefix + "_bctrl_quadmesh.obj"; 
    std::string spot_quadmesh_path  = "../../g1-quad-beziers/" + folder + "/" + prefix + "_bctrl_patches.obj"; 

    if (false){
            mesh_path               = "../../g1-quad-beziers/egg/out.obj";
            reference_mesh_path     = "../../g1-quad-beziers/egg/egg_final_tesselated.obj";
            our_cp_mesh_path        =  "../../g1-quad-beziers/egg/out_ours.obj"; 
            bctrl_quadmesh_path     =  "../../g1-quad-beziers/egg/egg_bctrl_quadmesh.obj"; 
            spot_quadmesh_path      =  "../../g1-quad-beziers/egg/egg_bctrl_patches.obj"; 
    }
    if (false){
            mesh_path               = "../../g1-quad-beziers/egg_simpler/out.obj";
            reference_mesh_path     = "../../g1-quad-beziers/egg_simpler/egg_simpler_final_tesselated.obj";
            our_cp_mesh_path        =  "../../g1-quad-beziers/egg_simpler/out_ours.obj"; 
            bctrl_quadmesh_path     =  "../../g1-quad-beziers/egg_simpler/egg_simpler_bctrl_quadmesh.obj"; 
            spot_quadmesh_path      =  "../../g1-quad-beziers/egg_simpler/egg_simpler_bctrl_patches.obj"; 
    }

    Eigen::MatrixXd Vbctrl;
    Eigen::MatrixXi Fbctrl;
    igl::readOBJ(bctrl_quadmesh_path, Vbctrl, Fbctrl);
    Eigen::MatrixXi Ebctrl;
    edge_network_from_quadmesh(Fbctrl,Ebctrl);

    Eigen::MatrixXd V_spot_quadmesh;
    Eigen::MatrixXi F_spot_quadmesh, E_spot_quadmesh;
    igl::readOBJ(spot_quadmesh_path, V_spot_quadmesh, F_spot_quadmesh);
    patch_edge_network_from_quadmesh(F_spot_quadmesh, E_spot_quadmesh);
    Eigen::MatrixXd Vpatches = V_spot_quadmesh( (Eigen::VectorXi) E_spot_quadmesh.reshaped(E_spot_quadmesh.rows()*2,1), Eigen::all);
    Eigen::MatrixXi Epatches(E_spot_quadmesh.rows(),2); for (int i=0; i<E_spot_quadmesh.rows(); i++) Epatches.row(i) = Eigen::RowVector2i(2*i,2*i+1);

	loadObj(mesh_path, mesh_init);
    igl::readOBJ(reference_mesh_path,Vr,Fr);

    std::cout << "Vr.shape: " << Vr.rows() << ", " << Vr.cols() << std::endl;
    std::cout << "Vr.block: " << Vr.block(Vr.rows()-1-12,0,12,3) << std::endl;
    bbx_diagonal = (Vr.colwise().maxCoeff() - Vr.colwise().minCoeff()).norm();
    std::cout << "bbx_diagonal: " << bbx_diagonal << std::endl;

    // igl::octree(Vr, tree.point_indices, tree.CH, tree.CN, tree.W);
    // std::cout << "... built octree" << std::endl;
    my_kd_tree_t t(3,std::cref(Vr),10);
    mytree = &t;
    std::cout << "... built kdtree" << std::endl;

    Eigen::MatrixXd Vours;
    Eigen::MatrixXi Fours;
    igl::readOBJ(our_cp_mesh_path,Vours,Fours);

    std::cout << "Vours.shape: " << Vours.rows() << ", " << Vours.cols() << std::endl;

    mesh_cur = mesh_init;
	subdivision_->loadMesh(mesh_cur);
    std::vector<std::vector<int>> F = transform_faces(mesh_cur);
    std::vector<std::vector<int>> F_init = transform_faces(mesh_init);

    // Eigen::MatrixXi E_subdiv;
    // edge_network_from_quadmesh(F_init,E_subdiv);

    polyscope::init();
    polyscope::view::upDir = polyscope::UpDir::ZUp;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
    polyscope::options::shadowBlurIters = 6;


    std::vector<polyscope::Structure *> slist;
    // K-Surface
    auto rs = polyscope::registerSurfaceMesh("Reference Surface", Vr, Fr);
    rs->setSurfaceColor(glm::vec3(227/255., 156/255., 38/255.));
    auto cp = polyscope::registerPointCloud("K-Surf Control Points", Vours);
    cp->setPointColor(glm::vec3(28/255., 110/255., 227/255.));
    slist.push_back((polyscope::Structure *) rs);
    slist.push_back((polyscope::Structure *) cp);
    structures.push_back(slist);

    slist.clear();
    slist.push_back(rs);
    slist.push_back(cp);
    auto sq = polyscope::registerCurveNetwork("Spot Quadmesh", Vpatches, Epatches)->setRadius(0.002);
    sq->setRadius(0.0015);
    slist.push_back((polyscope::Structure *) sq);
    structures.push_back(slist);

    // conversion: 
    auto bmc = polyscope::registerPointCloud("Bezier Middle Ctrl", mesh_init.positions);
    auto bac = polyscope::registerPointCloud("Bezier Ctrl Points", Vbctrl);
    bac->setPointRadius(0.003);
    bmc->setPointColor(glm::vec3(106/255., 28/255., 227/255.));
    auto bcm = polyscope::registerCurveNetwork("Bezier Control Mesh", Vbctrl, Ebctrl)->setRadius(0.001);
    bcm->setColor(bmc->getPointColor());

    slist.clear();
    slist.push_back(rs);
    slist.push_back(sq);
    slist.push_back((polyscope::Structure *) bcm);
    slist.push_back((polyscope::Structure *) bmc);
    slist.push_back((polyscope::Structure *) bac);
    structures.push_back(slist);


    // final subdivision surface
    slist.clear();
    slist.push_back(bmc);
    auto sds = polyscope::registerSurfaceMesh("Subdivision Surface", mesh_cur.positions, F);
    sds->setSurfaceColor(glm::vec3(28/255., 110/255., 227/255.));
    slist.push_back((polyscope::Structure *) sds);

    // auto scp = polyscope::registerCurveNetwork("SD control grid", mesh_init.positions, E_subdiv);
    // slist.push_back((polyscope::Structure *) scp);
    structures.push_back(slist);

    polyscope::state::userCallback = myCallback;
    polyscope::show();

	return 0;
}
