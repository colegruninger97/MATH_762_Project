#include <iostream>
#include <fstream>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include "AdvectionElementIntegration.hpp"

using namespace dealii;

/// The initial conditions for the mass
template<unsigned int dim>
class InitialConditionMass : public dealii::Function<dim> {
public:
  InitialConditionMass() = default;

  virtual ~InitialConditionMass() = default;

  virtual double value(dealii::Point<dim> const& p, const unsigned int component=0) const override;

private:
};

template<unsigned int dim>
double InitialConditionMass<dim>::value(dealii::Point<dim> const& p, const unsigned int component) const {
  (void)component;
  Assert(component==0, ExcIndexRange(component, 0, 1));
  if( p(1)<0.5 ) {
    return 0.5;
  }
  return 0.0;

  //return p(0);
}

/// The advection problem class
template<unsigned int dim>
class AdvectionProblem {
public:
  /// Default constructor
  AdvectionProblem();

  virtual ~AdvectionProblem() = default;

  void SetupSystem();

  void AssembleSystem();

  void Solve(dealii::Vector<double>& solution);

  void RefineGrid();

  void OutputResults(unsigned int const cycle) const;

  void Run();

private:

  /// Set up the initial condition
  /**
    @param[in] vec Set the intial condition onto this vector
  */
  void SetInitialCondition(dealii::Vector<double>& vec);

  /// Globally refine the mesh
  /**
    @param[in] nrefine The number of global refinements
  */
  void InitializeGrid(unsigned int const nrefine);

  void SetupDoFs();

  using DoFInfo = dealii::MeshWorker::DoFInfo<dim>;
  using CellInfo = dealii::MeshWorker::IntegrationInfo<dim>;

  /// Degree of the FEM basis
  const unsigned int femDegree = 2;

  dealii::Triangulation<dim> triangulation;
  const dealii::MappingQ1<dim> mapping;
  dealii::FE_DGQ<dim> fe;
  dealii::DoFHandler<dim> dofHandler;

  dealii::AffineConstraints<double> constraints;

  dealii::SparsityPattern sparsityPattern;
  dealii::SparseMatrix<double> systemMatrix;

  /// The solution to the advection equation at the current timestep
  dealii::Vector<double> solution;

  /// The solution to the advection equation at the previous timestep
  dealii::Vector<double> old_solution;

  dealii::Vector<double> rhs;

  /// The length of each timestep
  const double timeStep = 1.0/1000.0;

  /// The current time
  double currTime;

  /// The current timestep
  unsigned int step = 0;

  /// The final time
  const double finalTime = 1.0;

  /// The maximum number of refinement levels
  const unsigned int maxGridLevels = 8;
};

template<unsigned int dim>
AdvectionProblem<dim>::AdvectionProblem() : mapping(), fe(femDegree), dofHandler(triangulation), currTime(timeStep) {
  // initialize a square grid and refine global
  InitializeGrid(maxGridLevels-1);

  // set up the system on this initial grid
  SetupSystem();

  // reset the old solution
  old_solution.reinit(dofHandler.n_dofs());
}

template<unsigned int dim>
void AdvectionProblem<dim>::SetupDoFs() {
  // distribute the dofs
  dofHandler.distribute_dofs(fe);
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dofHandler, constraints);
  constraints.close();
}

template<unsigned int dim>
void AdvectionProblem<dim>::SetupSystem() {
  // set up the degrees of freedom for the system
  SetupDoFs();

  // generate a sparsity pattern
  DynamicSparsityPattern dsp(dofHandler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dofHandler, dsp, constraints);
  sparsityPattern.copy_from(dsp);

  // set up the strucutre of the components
  systemMatrix.clear();
  systemMatrix.reinit(sparsityPattern);
  solution.reinit(dofHandler.n_dofs());
  rhs.reinit(dofHandler.n_dofs());
}

template<unsigned int dim>
void AdvectionProblem<dim>::SetInitialCondition(Vector<double>& vec) {
  VectorTools::project(dofHandler, constraints, QGauss<dim>(fe.degree+1), InitialConditionMass<dim>(), vec);
}

template<unsigned int dim>
void AdvectionProblem<dim>::AssembleSystem() {
  // an object that knows about data structures and local integration
  MeshWorker::IntegrationInfoBox<dim> infoBox;

  // initialize the quadrature formula
  const unsigned int nGauss = dofHandler.get_fe().degree + 1;
  infoBox.initialize_gauss_quadrature(nGauss, nGauss, nGauss);

  // these are the values we need to integrate our system
  infoBox.initialize_update_flags();
  const UpdateFlags updateFlags = update_quadrature_points | update_values | update_gradients | update_JxW_values;
  infoBox.add_update_flags(updateFlags, true, true, true, true);

  // initialize the finite element values
  infoBox.initialize(fe, mapping);

  // create an object that forwards local data to the assembler
  MeshWorker::DoFInfo<dim> dofInfo(dofHandler);

  // create the assembler---tell it where to put local data
  MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double> > assembler;
  assembler.initialize(systemMatrix, rhs);

  //IntegrateCell<dim> cellIntegration(old_solution, currTime, timeStep);
  AdvectionCellIntegration<dim> cellIntegration(old_solution, currTime, timeStep);
  AdvectionBoundaryIntegration<dim> boundaryIntegration(old_solution, currTime, timeStep);
  AdvectionFaceIntegration<dim> faceIntegration(old_solution, currTime, timeStep);

  MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim> >(dofHandler.begin_active(), dofHandler.end(), dofInfo, infoBox, cellIntegration, boundaryIntegration, faceIntegration, assembler);
}

template<unsigned int dim>
void AdvectionProblem<dim>::Solve(Vector<double>& solution) {
  SolverControl solverControl(100, 1.0e-12);
  SolverRichardson<> solver(solverControl);

  PreconditionBlockSSOR<SparseMatrix<double> > preconditioner;
  preconditioner.initialize(systemMatrix, fe.dofs_per_cell);

  solver.solve(systemMatrix, solution, rhs, preconditioner);
}

template<unsigned int dim>
void AdvectionProblem<dim>::InitializeGrid(unsigned int const nrefine) {
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(nrefine);
}

template<unsigned int dim>
void AdvectionProblem<dim>::RefineGrid() {
  std::cout << "\tsolution size before refinement: " << solution.size() << std::endl;

  // save some memory and use floats because we just need to estimate the erros

  // estimate the error in each cell
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate(dofHandler, QGauss<dim-1>(femDegree+1), {}, solution, estimated_error_per_cell);

  // refine cells with the largest estimated error (80 per cent of the error) and coarsens those cells with the smallest error (10 per cent of the error)
  GridRefinement::refine_and_coarsen_fixed_fraction(triangulation, estimated_error_per_cell, 0.8, 0.1);

  // to prevent the decrease in time step size limit the maximal refinement depth of the meshlimit the maximal refinement depth of the mesh
  if( triangulation.n_levels()>maxGridLevels ) {
    for( auto& cell : triangulation.active_cell_iterators_on_level(maxGridLevels) ) { cell->clear_refine_flag(); }
  }

  // transfer the solution vectors from the old mesh to the new one
  old_solution = solution;
  SolutionTransfer<dim> transfer(dofHandler);

  triangulation.prepare_coarsening_and_refinement();
  transfer.prepare_for_coarsening_and_refinement(old_solution);

  // prefore the refinement and reset the DoFs
  triangulation.execute_coarsening_and_refinement();
  SetupSystem();

  // move the solution onto the old solution
  //solution.reinit(dofHandler.n_dofs());
  transfer.interpolate(old_solution, solution);
  //solution = old_solution;

  constraints.distribute(solution);
  constraints.distribute(old_solution);

  std::cout << "\tsolution size after refinement: " << solution.size() << std::endl;
}

template<unsigned int dim>
void AdvectionProblem<dim>::OutputResults(unsigned int const cycle) const {
  { // write the grid in eps format
    const std::string filename = "grid-" + std::to_string(cycle) + ".eps";
    deallog << "Writing grid to <" << filename << ">" << std::endl;
    std::ofstream epsOutput(filename);
    GridOut gridOut;
    gridOut.write_eps(triangulation, epsOutput);
  }
  { // output the solution
    DataOut<dim> dataOut;
    dataOut.attach_dof_handler(dofHandler);
    dataOut.add_data_vector(solution, "u");
    dataOut.build_patches();
    const std::string filename = "sol-" + std::to_string(cycle) + ".vtu";
    deallog << "Writing solution to <" << filename << ">" << std::endl;
    DataOutBase::VtkFlags vtkFlags;
    vtkFlags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    dataOut.set_flags(vtkFlags);
    std::ofstream output(filename);
    dataOut.write_vtu(output);
  }
}

template<unsigned int dim>
void AdvectionProblem<dim>::Run() {
  // project the initial conditions---on whatever grid is currently setup (constructor initializes something)
  SetInitialCondition(old_solution);

  // outptu the initial conditions
  solution = old_solution;
  OutputResults(step++);

  // loop through time
  for( ; currTime<finalTime+timeStep/2.0; currTime+=timeStep, ++step ) {
    std::cout << "time: " << currTime << " of " << finalTime << std::endl;

    // set up the sparsity pattern for the system and reinit the solution
    SetupSystem();

    // assemble the system
    AssembleSystem();

    // solve the system
    Solve(solution);

    // refine the grid
    RefineGrid();

    // reset the old solution
    old_solution = solution;

    //std::cout << "time: " << currTime << " of " << finalTime << ", step: " << step << std::endl;
    OutputResults(step);
  }
}

int main(int argc, char **argv) {
  AdvectionProblem<2> dgmethod;
  dgmethod.Run();
}
