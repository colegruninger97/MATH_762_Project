#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/base/tensor_function.h>
#include <iostream>
#include <fstream>
namespace Stress_Evolution
{
using namespace dealii;

//template <int dim>
//class InitialValues : public Function<dim>
//{
//public:
//  InitialValues()
//    : Function<dim>(dim*dim)
//  {}
//    virtual void vector_value(const Point<dim> &p, Vector<double> & values) const override;
//};
//
//template <int dim>
//void InitialValues<dim>::vector_value(const Point<dim> &p, Vector<double> & values) const
//{
//
//    for(unsigned int i = 0; i<dim*dim; i++)
//    {
//        values(i) = 0.0; //Setup zero initial data
//    }
//
//};
//
//template <int dim>
//class InflowConditions : public Function<dim>
//{
//public:
//    InflowConditions()
//    : Function<dim>(1)
//    {}
//    virtual void vector_value(const Point<dim> &p, Vector<double> & values) const override;
//
//};
//
//template <int dim>
//void InflowConditions<dim>::vector_value(const Point<dim> &p, Vector<double> & values) const
//{
//
//    //Inflow Boundary Conditions for pipe flow according to exact solution ----> only dimension=2 works :'(
//    //Need to setup a Boundary ID
//
//    double lambda = 0.01;
//    double mu = 1.0;
//    double mu_p = 1.0;
//    double P_x = -1.0;
//
//    values(0) = 2*lambda*mu_p*((P_x/mu)*p[1] - (P_x/(2*mu)))*((P_x/mu)*p[1] - (P_x/(2*mu))); //tau_xx
//    values(1) = mu_p*((P_x/mu)*p[1] - (P_x/(2*mu)));
//    values(2) = values(1);
//    values(3) = 0.0;
//
//};
template <int dim>
class InitialValues : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
        return 0.0;
    }
    
    
};

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
};

template<int dim>
double BoundaryValues<dim>::value(const Point<dim> & p,
                                  const unsigned int /*component*/) const
{
    
    double lambda = 0.01;
    double mu = 1.0;
    double mu_p = 1.0;
    double P_x = -1.0;
    return 1.0;//2*lambda*mu_p*((P_x/(mu))*p[1] - P_x*(1/(2*mu)))*((P_x/(mu))*p[1] - P_x*(1/(2*mu)));
    
};



//Velocity field associated with pipe flow
template <int dim>
Tensor<1, dim> Velocity(const Point<dim> &p)
{
    double mu = 1.0;
    double P_x = -1.0;
    Tensor<1, dim> pipe_flow;
    pipe_flow[0] = 1.0;//P_x*(1/(2*mu))*p[1]*(p[1]-1.0);
    pipe_flow[1] = 0.0;
    return pipe_flow;
}

template <int dim>
class AdvectionProblem
{
public:
    AdvectionProblem();
    void run();
    
private:
    void setup_system();
    void assemble_rhs();
    void solve();
    void refine_grid();
    void output_results() const;
    
    
    Triangulation<dim>   triangulation;
    const MappingQ1<dim> mapping;
    
    // Furthermore we want to use DG elements.
    const FE_DGQ<dim> fe;
    DoFHandler<dim> dof_handler;
    
    AffineConstraints<double> constraints;
    
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> matrix_tau;
    double time_step;
    double time;
    unsigned int timestep_number;
    Vector<double> solution;
    Vector<double> right_hand_side;
    Vector<double> old_solution;
};


// We start with the constructor. The 1 in the constructor call of
// <code>fe</code> is the polynomial degree.
template <int dim>
AdvectionProblem<dim>::AdvectionProblem()
: mapping()
, fe(1)
, dof_handler(triangulation)
, time(0.0)
, timestep_number(0)
{}

template<int dim>
void AdvectionProblem<dim>::setup_system()
{
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(3);
    time_step = 0.01*(triangulation.begin_active()->diameter())/(std::sqrt(2));
    
    dof_handler.distribute_dofs(fe);
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,dsp);
    sparsity_pattern.copy_from(dsp);
    
    mass_matrix.reinit(sparsity_pattern);
    matrix_tau.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    right_hand_side.reinit(dof_handler.n_dofs());
    
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree+1), mass_matrix);
    
    constraints.close();
    
    
    //Setup a Boundary ID for the prescribed inflow data
    for (auto &face : triangulation.active_face_iterators()){
        if (face->at_boundary())
        {
            if (face->center()[0] == 0)
            {
                face->set_boundary_id (1);
            }
        }
    }
};


template<int dim>
void AdvectionProblem<dim>::assemble_rhs()
{
    QGauss<dim> quadrature(fe.degree + 2);
    QGauss<dim-1> quadrature_face(fe.degree + 2);
    FEValues<dim> fe_values(fe, quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,quadrature_face, update_values | update_gradients | update_quadrature_points | update_normal_vectors | update_JxW_values);
    FEFaceValues<dim> fe_face_values_neighbor(fe,quadrature_face,update_values | update_quadrature_points | update_JxW_values);
    
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    
    const unsigned int n_q_points = quadrature.size();
    const unsigned int n_face_q_points = quadrature_face.size();
    Vector<double>      local_rhs(dofs_per_cell);
    
    
    
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    
    //Get old solution values at the faces
    
    std::vector<Vector<double>> old_solution_values(n_q_points,Vector<double>(1));
    
    
    //Loop over the cells
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(old_solution,old_solution_values);
        const auto &q_points = fe_values.get_quadrature_points();
        local_rhs = 0;
        
        for (unsigned int q = 0; q<n_q_points; ++q)
        {
            const double old_s = old_solution_values[q](0);
            auto Velocity_q = Velocity(q_points[q]);
            for (unsigned int i = 0; i<dofs_per_cell; ++i)
            {
                local_rhs(i) += Velocity_q
                * fe_values.shape_grad(i,q)
                * old_s
                * time_step
                * fe_values.JxW(q);
            }
        }
        
        
        
        
        //Now loop over all the faces to get the int_{\partial K} u (dot) n tau^n phi\,dS term
        
        // get the old solution values along the faces
        std::vector<Vector<double>> old_solution_face_values(n_face_q_points,Vector<double>(1));
        std::vector<Vector<double>> old_neighbor_solution_face_values(n_face_q_points,Vector<double>(1));
        
        
        
        
        
        
        for (const auto face_n : cell->face_indices())
        {
            fe_face_values.reinit(cell,face_n);
            fe_face_values.get_function_values(old_solution, old_solution_face_values);
            
            if(cell->at_boundary(face_n))
            {
                const auto &q_face_points = fe_face_values.get_quadrature_points();
                const unsigned int n_face_dofs = fe_face_values.get_fe().n_dofs_per_cell();
                const std::vector<Tensor<1,dim>> &normals = fe_face_values.get_normal_vectors();
                BoundaryValues<dim> BC_function;
                
                if(cell->face(face_n)->boundary_id()==1)
                {
                    
                    for(unsigned int q = 0; q<n_face_q_points; q++){
                        
                        double Velocity_dot_n = Velocity(q_face_points[q]) * normals[q];
                        for(unsigned int i = 0; i<dofs_per_cell;++i){
                            local_rhs(i) -= time_step
                            * BC_function.value(q_face_points[q])
                            * fe_face_values.shape_value(i,q)
                            * Velocity_dot_n
                            * fe_face_values.JxW(q);
                        }
                        
                    }
                }
                else{
                    for(unsigned int q = 0; q<n_face_q_points;q++){
                        
                        double Velocity_dot_n = Velocity(q_face_points[q])*normals[q];
                        double old_s_face = old_solution_face_values[q](0);
                        for(unsigned int i=0; i<dofs_per_cell;++i){
                            local_rhs(i) -= time_step
                            * old_s_face
                            * fe_face_values.shape_value(i,q)
                            * Velocity_dot_n
                            * fe_face_values.JxW(q);
                            
                        }
                    }
                    
                }
            }
            else{
                const auto neighbor = cell->neighbor(face_n);
                const unsigned int neighbor_face = cell->neighbor_of_neighbor(face_n);
                fe_face_values_neighbor.reinit(neighbor,neighbor_face);
                fe_face_values_neighbor.get_function_values(old_solution,old_neighbor_solution_face_values);
                Point<dim> neighbor_q_point = fe_face_values_neighbor.get_quadrature_points()[0];
                const auto &q_face_points = fe_face_values.get_quadrature_points();
                const unsigned int n_face_dofs = fe_face_values.get_fe().n_dofs_per_cell();
                const std::vector<Tensor<1,dim>> &normals = fe_face_values.get_normal_vectors();
                
                
                Assert(fe_face_values.get_quadrature_points()[0] == neighbor_q_point, ExcMessage("q points should match"));
                
                
                
                for(unsigned int q = 0; q<n_face_q_points; q++)
                {
                    const double old_s_face = old_solution_face_values[q](0);
                    const double old_neighbor_s = old_neighbor_solution_face_values[q](0);
                    double Velocity_dot_n = Velocity(q_face_points[q]) * normals[q];
                    if(Velocity_dot_n<0){
                        for(unsigned int i = 0; i<dofs_per_cell; ++i){
                            local_rhs(i) -= time_step
                            * old_neighbor_s
                            * fe_face_values.shape_value(i,q)
                            * Velocity_dot_n
                            * fe_face_values.JxW(q);
                            
                        }
                    }
                    else{
                        for(unsigned int i = 0; i<dofs_per_cell; ++i){
                            local_rhs(i) -= time_step
                            * old_s_face
                            * Velocity_dot_n
                            * fe_face_values.shape_value(i,q)
                            * fe_face_values.JxW(q);
                        }
                        
                        
                    }
                }
                
            }
            
            
            
            
        }
        
        //Finish building the RHS
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i=0;i<dofs_per_cell;++i){
            right_hand_side(local_dof_indices[i]) += local_rhs(i);
        }
    }
};




template <int dim>
void AdvectionProblem<dim>::solve()
{
    SolverControl                    solver_control(1000, 1e-8); //Use CG since the mass matrix is SPD
    SolverCG<Vector<double>>         solver(solver_control);
    
    solver.solve(matrix_tau,solution, right_hand_side, PreconditionIdentity());
    
    
    std::cout << "  Solver converged in " << solver_control.last_step()
    << " iterations." << std::endl;
}


template <int dim>
void AdvectionProblem<dim>::output_results() const
{
    
    const std::string filename = "solution-" + Utilities::int_to_string(timestep_number,3) + ".vtk";
    std::cout << "  Writing solution to <" << filename << ">" << std::endl;
    std::ofstream output(filename);
    
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "Tau", DataOut<dim>::type_dof_data);
   // data_out.add_data_vector(right_hand_side, "RHS", DataOut<dim>::type_dof_data);
    
    data_out.build_patches(mapping);
    
    data_out.write_vtk(output);
    
    //  {
    //    Vector<float> values(triangulation.n_active_cells());
    //    VectorTools::integrate_difference(dof_handler,
    //                                      solution,
    //                                      ZeroFunction<dim>(),
    //                                      values,
    //                                      QGauss<dim>(fe.degree + 1),
    //                                      VectorTools::Linfty_norm);
    //    const double l_infty =
    //      VectorTools::compute_global_error(triangulation,
    //                                        values,
    //                                        VectorTools::Linfty_norm);
    //    std::cout << "  L-infinity norm: " << l_infty << std::endl;
    //  }
}


//The run function
template <int dim>
void AdvectionProblem<dim>::run()
{
    
    setup_system();
    Vector<double> tmp1;
    Vector<double> tmp2;
    Vector<double> tmp0;
    Vector<double> tmp3;
    Vector<double> tmp4;
    Vector<double> tmp5;
    matrix_tau.copy_from(mass_matrix);
    tmp1.reinit(solution.size());
    tmp0.reinit(solution.size());
    tmp2.reinit(solution.size());
    tmp3.reinit(solution.size());
    tmp4.reinit(solution.size());
    tmp5.reinit(solution.size());
    //use the inital conditions for the old_solution
    {
        VectorTools::project(dof_handler,
                             constraints,
                             QGauss<dim>(fe.degree+2),
                             InitialValues<dim>(),
                             solution);
        old_solution = solution;
        output_results();
        ++timestep_number;
        time+=time_step;
        
    }
    //Below is my attempt at implementing a third order TVD Runge Kutta Scheme taken from Gottlieb and Shu's paper https://www.cfm.brown.edu/people/sg/SSP1.pdf
    for(; time<=0.6; time+= time_step, ++timestep_number)
    {
        assemble_rhs();
        mass_matrix.vmult_add(right_hand_side,old_solution);
        tmp0 = old_solution;
        solve();
        old_solution = solution;
        tmp1 = solution;
        tmp1 *= 0.25;
        tmp0 *= 0.75;
        right_hand_side = 0;
        assemble_rhs();
        right_hand_side *= 0.25;
        mass_matrix.vmult_add(right_hand_side,tmp1);
        mass_matrix.vmult_add(right_hand_side,tmp0);
        solve();
        right_hand_side = 0;
        old_solution = solution;
        tmp2 = solution;
        tmp2 *= (2.0/3.0);
        tmp0 *= (4.0/9.0);
        assemble_rhs();
        right_hand_side *= (2.0/3.0);
        mass_matrix.vmult_add(right_hand_side,tmp2);
        mass_matrix.vmult_add(right_hand_side,tmp0);
        solve();
        output_results();
        old_solution = solution;
        right_hand_side = 0;
        
    }
        
        
//        assemble_rhs();
//        right_hand_side = 0; //Set to zero for testing purposes
//
//        //Checking that the TVD scheme works for u_t = -u;
//        mass_matrix.vmult_add(right_hand_side,old_solution);
//        tmp0 = old_solution; //u^n
//        tmp1 = tmp0;
//        tmp1 *= -time_step;
//        mass_matrix.vmult_add(right_hand_side,tmp1);
//        solve();
//        right_hand_side = 0;
//        tmp2 = solution;
//        tmp3 = tmp2;
//        tmp2 *= 0.25;
//        tmp3 *= -0.25*time_step;
//        tmp0 *= (3.0/4.0);
//        mass_matrix.vmult_add(right_hand_side,tmp3);
//        mass_matrix.vmult_add(right_hand_side,tmp2);
//        mass_matrix.vmult_add(right_hand_side,tmp0);
//        solve();
//        right_hand_side = 0;
//        tmp4 = solution;
//        tmp5 = solution;
//        tmp5 *= (2.0/3.0);
//        tmp4 *= -(2.0/3.0)*time_step;
//        tmp0 *= (4.0/9.0);
//        mass_matrix.vmult_add(right_hand_side,tmp4);
//        mass_matrix.vmult_add(right_hand_side,tmp5);
//        mass_matrix.vmult_add(right_hand_side,tmp0);
//        solve();
//        old_solution = solution;
//        output_results();
    
    
}

}
// namespace Stress_Evolution

int main()
{
    try
    {   using namespace Stress_Evolution;
        AdvectionProblem<2> dgmethod;
        dgmethod.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
        << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
        << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    
    return 0;
}

