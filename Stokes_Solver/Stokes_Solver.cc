//Header files
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

//UMFPACK
#include <deal.II/lac/sparse_direct.h>


#include <deal.II/lac/sparse_ilu.h>


#include <iostream>
#include <fstream>
#include <memory>

// As in all programs, the namespace dealii is included:
namespace Stokes_Solver
{
  using namespace dealii;


  template <int dim>
  struct InnerPreconditioner;

  // 2D preconditioner
  template <>
  struct InnerPreconditioner<2>
  {
    using type = SparseDirectUMFPACK;
  };

  // 3D preconditioner
  template <>
  struct InnerPreconditioner<3>
  {
    using type = SparseILU<double>;
  };

//The preconditoners above are chosen based off of step 22

  template <int dim>
  class StokesSolve
  {
  public:
    StokesSolve(const unsigned int degree); //constructor
    void run();

  private:
    void setup_dofs();
    void assemble_system();
    void solve();
    void output_results();
    void refine_mesh();

    const unsigned int degree;

    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> constraints;

    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockSparsityPattern      preconditioner_sparsity_pattern;
    BlockSparseMatrix<double> preconditioner_matrix;

    BlockVector<double> solution;
    BlockVector<double> system_rhs;

    std::shared_ptr<typename InnerPreconditioner<dim>::type> preconditioner;
  };

//Exact solution as specified by me. 4-roll mill
template <int dim>
class ExactSolution : public Function<dim>
{
public: ExactSolution()
    : Function<dim>(dim+1)
    {}
    
    virtual void vector_value(const Point<dim> &p, Vector<double> & values) const override;
};

template <int dim>
void ExactSolution<dim>::vector_value(const Point<dim> & p, Vector<double> & values) const
{
    const double pi = numbers::PI;
    Assert(values.size() == dim+1,
           ExcDimensionMismatch(values.size(), dim+1));
    values(0) = std::sin(2*pi*p(0))*std::cos(2*pi*p(1));
    values(1) = -1.0*std::cos(2*pi*p(0))*std::sin(2*pi*p(1)); //This choice is obviously dependent on the unit square domain
    values(2) = 0;
}


//Setup the RHS -------------------------------------------------------------------------------------------------------------------------------------------//
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide()
      : Function<dim>(dim + 1)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  value) const override;
  };



  template <int dim>
  void RightHandSide<dim>::vector_value(const Point<dim> &p,
                                        Vector<double> &  values) const
  {
      const double pi = numbers::PI;
      const double mu = 1.0;//viscosity is here for now
      values(0) = 1.0*mu*8*pi*pi*std::sin(2*pi*p(0))*std::cos(2*pi*p(1));
      values(1) = -1.0*mu*8*pi*pi*std::cos(2*pi*p(0))*std::sin(2*pi*p(1));
      values(2) = 0; //This is 4-roll mill forcing ...
  }

//---------------------------------------------------------------------------------------------------------------------------------------------------------//
  template <class MatrixType, class PreconditionerType>
  class InverseMatrix : public Subscriptor
  {
  public:
    InverseMatrix(const MatrixType &        m,
                  const PreconditionerType &preconditioner);

    void vmult(Vector<double> &dst, const Vector<double> &src) const;

  private:
    const SmartPointer<const MatrixType>         matrix;
    const SmartPointer<const PreconditionerType> preconditioner;
  };


  template <class MatrixType, class PreconditionerType>
  InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
    const MatrixType &        m,
    const PreconditionerType &preconditioner)
    : matrix(&m)
    , preconditioner(&preconditioner)
  {}


  template <class MatrixType, class PreconditionerType>
  void InverseMatrix<MatrixType, PreconditionerType>::vmult(
    Vector<double> &      dst,
    const Vector<double> &src) const
  {
    SolverControl            solver_control(src.size(), 1e-6 * src.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    dst = 0;

    cg.solve(*matrix, dst, src, *preconditioner);
  }


  template <class PreconditionerType>
  class SchurComplement : public Subscriptor
  {
  public:
    SchurComplement(
      const BlockSparseMatrix<double> &system_matrix,
      const InverseMatrix<SparseMatrix<double>, PreconditionerType> & Inverse);

    void vmult(Vector<double> &dst, const Vector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
    const SmartPointer<
      const InverseMatrix<SparseMatrix<double>, PreconditionerType>>
      Inverse;

    mutable Vector<double> tmp1, tmp2;
  };



  template <class PreconditionerType>
  SchurComplement<PreconditionerType>::SchurComplement(
    const BlockSparseMatrix<double> &system_matrix,
    const InverseMatrix<SparseMatrix<double>, PreconditionerType> & Inverse)
    : system_matrix(&system_matrix)
    , Inverse(&Inverse)
    , tmp1(system_matrix.block(0, 0).m())
    , tmp2(system_matrix.block(0, 0).m())
  {}


  template <class PreconditionerType>
  void
  SchurComplement<PreconditionerType>::vmult(Vector<double> &      dst,
                                             const Vector<double> &src) const
  {
    system_matrix->block(0, 1).vmult(tmp1, src);
    Inverse->vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
  }


  template <int dim>
  StokesSolve<dim>::StokesSolve(const unsigned int degree)
    : degree(degree)
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1) //Taylor Hood finite elements
    , dof_handler(triangulation)
  {}


  template <int dim>
  void StokesSolve<dim>::setup_dofs()
  {
    preconditioner.reset();
    system_matrix.clear();
    preconditioner_matrix.clear();

    dof_handler.distribute_dofs(fe);

    std::vector<unsigned int> block_component(dim + 1, 0); //Need components for velocities and the pressure
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

 //Setup the boundary conditions for the velocity
    {
      constraints.clear();

      FEValuesExtractors::Vector velocities(0);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               ExactSolution<dim>(), //Use the prescribed solution on the boundary
                                               constraints,
                                               fe.component_mask(velocities)); //Dirichlet Boundary conditions for the velocities only, no BCs are required for the pressure.
    }

    constraints.close();

//Count the dofs componentwise using the block structure.
    const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    std::cout << "   Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_p << ')' << std::endl;

  //Figure out the sparsity pattern...
    {
      BlockDynamicSparsityPattern dsp(2, 2);

      dsp.block(0, 0).reinit(n_u, n_u);
      dsp.block(1, 0).reinit(n_p, n_u);
      dsp.block(0, 1).reinit(n_u, n_p);
      dsp.block(1, 1).reinit(n_p, n_p);

      dsp.collect_sizes();

      Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);

      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if (!((c == dim) && (d == dim)))
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;

      DoFTools::make_sparsity_pattern(
        dof_handler, coupling, dsp, constraints, false);

      sparsity_pattern.copy_from(dsp);
    }

    {
      BlockDynamicSparsityPattern preconditioner_dsp(2, 2);

      preconditioner_dsp.block(0, 0).reinit(n_u, n_u);
      preconditioner_dsp.block(1, 0).reinit(n_p, n_u);
      preconditioner_dsp.block(0, 1).reinit(n_u, n_p);
      preconditioner_dsp.block(1, 1).reinit(n_p, n_p);

      preconditioner_dsp.collect_sizes();

      Table<2, DoFTools::Coupling> preconditioner_coupling(dim + 1, dim + 1);

      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if (((c == dim) && (d == dim)))
            preconditioner_coupling[c][d] = DoFTools::always;
          else
            preconditioner_coupling[c][d] = DoFTools::none;

      DoFTools::make_sparsity_pattern(dof_handler,
                                      preconditioner_coupling,
                                      preconditioner_dsp,
                                      constraints,
                                      false);

      preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
    }

    // Create system matrix and right hand side using the block structure
    system_matrix.reinit(sparsity_pattern);
    preconditioner_matrix.reinit(preconditioner_sparsity_pattern);

    solution.reinit(2);
    solution.block(0).reinit(n_u);
    solution.block(1).reinit(n_p);
    solution.collect_sizes();

    system_rhs.reinit(2);
    system_rhs.block(0).reinit(n_u);
    system_rhs.block(1).reinit(n_p);
    system_rhs.collect_sizes();
  }

//Assemble the linear system
  template <int dim>
  void StokesSolve<dim>::assemble_system()
  {
    system_matrix         = 0;
    system_rhs            = 0;
    preconditioner_matrix = 0;

    QGauss<dim> quadrature_formula(degree + 2);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                   dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const RightHandSide<dim>    right_hand_side;
    std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim + 1));

//Use FEValuesExtractors to obtain velocities and pressures. This is basically a fancy way of indexing...
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);


   std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<double>                  phi_p(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        local_matrix                = 0;
        local_preconditioner_matrix = 0;
        local_rhs                   = 0;

        right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                          rhs_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                  for (unsigned int j = 0; j<dofs_per_cell; ++j)
                  {
                      local_matrix(i,j) += (scalar_product(fe_values[velocities].gradient(i,q), fe_values[velocities].gradient(j,q))
                                         - fe_values[velocities].divergence(i,q) * fe_values[pressure].value(j,q))
                                         * fe_values.JxW(q);
                      
                      local_preconditioner_matrix(i,j) +=
                        (fe_values[pressure].value(i,q) * fe_values[pressure].value(j,q))
                      * fe_values.JxW(q);
                      
                  }
                  const unsigned int component_i =
                  fe.system_to_component_index(i).first;
                  local_rhs(i) += (fe_values.shape_value(i, q)   // (phi_u_i(x_q)
                                   * rhs_values[q](component_i)) // * f(x_q))
                                  * fe_values.JxW(q);            // * dx
              }
          }
          
        //distribute the constraints
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_matrix,
                                               local_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
        constraints.distribute_local_to_global(local_preconditioner_matrix,
                                               local_dof_indices,
                                               preconditioner_matrix);
      }

   //Generate a preconditioner for the velocity portion of the matrix
    std::cout << "   Computing preconditioner..." << std::endl << std::flush;

    preconditioner =
      std::make_shared<typename InnerPreconditioner<dim>::type>();
    preconditioner->initialize(
      system_matrix.block(0, 0),
      typename InnerPreconditioner<dim>::type::AdditionalData());
  }


//Solve the resulting linear system ----> Using what's given to me in step 22 here...
  template <int dim>
  void StokesSolve<dim>::solve()
  {
    const InverseMatrix<SparseMatrix<double>,
                        typename InnerPreconditioner<dim>::type>
                   Inverse(system_matrix.block(0, 0), *preconditioner);
    Vector<double> tmp(solution.block(0).size());
    {
      Vector<double> schur_rhs(solution.block(1).size());
      Inverse.vmult(tmp, system_rhs.block(0));
      system_matrix.block(1, 0).vmult(schur_rhs, tmp);
      schur_rhs -= system_rhs.block(1);

      SchurComplement<typename InnerPreconditioner<dim>::type> schur_complement(
        system_matrix, Inverse);

      // The usual control structures for the solver call are created...
      SolverControl            solver_control(solution.block(1).size(),
                                   1e-6 * schur_rhs.l2_norm());
      SolverCG<Vector<double>> cg(solver_control);

      //Create a preconditioner for the Schur complement
      SparseILU<double> preconditioner;
      preconditioner.initialize(preconditioner_matrix.block(1, 1),
                                SparseILU<double>::AdditionalData());

      InverseMatrix<SparseMatrix<double>, SparseILU<double>> m_inverse(
        preconditioner_matrix.block(1, 1), preconditioner);

      //conjugate gradient solve
      cg.solve(schur_complement, solution.block(1), schur_rhs, m_inverse);
      //distribute the constraints
      constraints.distribute(solution);

      std::cout << "  " << solver_control.last_step()
                << " outer CG Schur complement iterations for pressure"
                << std::endl;
    }

    //Finally, solve for the velocity...
    {
      system_matrix.block(0, 1).vmult(tmp, solution.block(1));
      tmp *= -1;
      tmp += system_rhs.block(0);

      Inverse.vmult(solution.block(0), tmp);

      constraints.distribute(solution);
    }
  }






  template <int dim>
  void StokesSolve<dim>::run()
  {
    {//setup the grid rq

        GridGenerator::hyper_cube(triangulation); //Solve on the unit square (cube) for now ....
        
    }


     triangulation.refine_global(5); //Solve on a 32 by 32 grid for now
      
      setup_dofs();
      
      std::cout<< " Assembling ..."<<std::endl <<std::flush;
      
      assemble_system();
      
      std::cout<< "Time to solve...." << std::flush;
      
      solve();
      
      output_results();
      
      std::cout<< "finished!"<<std::endl;

  }

//output the results to visualize in Visit....
  template <int dim>
  void
  StokesSolve<dim>::output_results()

  {
     // solution.block(dim-1)  = 0.0; //Here setting the pressure equal to its exact solution
      //Compute the pointwise maximum error
      Vector<double> max_error_per_cell(triangulation.n_active_cells());
        {
          MappingQGeneric<dim> mapping(1);
          VectorTools::integrate_difference(mapping,
                                            dof_handler,
                                            solution,
                                            ExactSolution<dim>(),
                                            max_error_per_cell,
                                            QIterated<dim>(QGauss<1>(2), 2),
                                            VectorTools::NormType::Linfty_norm);
          std::cout << "maximum error = " << *std::max_element(max_error_per_cell.begin(),
                                                               max_error_per_cell.end())
                    << std::endl;
      }

    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.add_data_vector(max_error_per_cell, "max_error_per_cell");
    data_out.build_patches();

    std::ofstream output(
      "solution_32.vtk");
    data_out.write_vtk(output);
  }
} // namespace Stokes_Solver



int main()
{
 try
    {
      using namespace Stokes_Solver;

      StokesSolve<2> Stokesflow(1); //I'm almost certain my setup only applies to the 2D case... not sure how to generalize rn...
        Stokesflow.run();
    }
    //Adding this in as per Bangerth's recommendations made in lecture videos/turtorials
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


