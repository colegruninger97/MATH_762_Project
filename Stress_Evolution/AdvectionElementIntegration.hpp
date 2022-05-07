#ifndef ADVECTIONELEMENTINTEGRATION_HPP_
#define ADVECTIONELEMENTINTEGRATION_HPP_

#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>

template<unsigned int dim>
class AdvectionElementIntegration {
public:
  AdvectionElementIntegration(dealii::Vector<double> const& old_solution, double const currTime, double const timeStep);

  virtual ~AdvectionElementIntegration() = default;

  virtual inline void operator()(dealii::MeshWorker::DoFInfo<dim>& dinfo, dealii::MeshWorker::IntegrationInfo<dim>& info) const {}

  virtual inline void operator()(dealii::MeshWorker::DoFInfo<dim>& dinfo1, dealii::MeshWorker::DoFInfo<dim>& dinfo2, dealii::MeshWorker::IntegrationInfo<dim>& info1, dealii::MeshWorker::IntegrationInfo<dim>& info2) const {}

protected:
  /// Store a constant reference to the old solution
  const dealii::Vector<double>& old_solution;

  /// The current time
  const double currTime;

  /// The timestep size
  const double timeStep;
private:
};

template<unsigned int dim>
class AdvectionCellIntegration : public AdvectionElementIntegration<dim> {
public:
  AdvectionCellIntegration(dealii::Vector<double> const& old_solution, double const currTime, double const timeStep);

  virtual ~AdvectionCellIntegration() = default;

  virtual void operator()(dealii::MeshWorker::DoFInfo<dim>& dinfo, dealii::MeshWorker::IntegrationInfo<dim>& info) const override;
private:
};

template<unsigned int dim>
class AdvectionBoundaryIntegration : public AdvectionElementIntegration<dim> {
public:
  AdvectionBoundaryIntegration(dealii::Vector<double> const& old_solution, double const currTime, double const timeStep);

  virtual ~AdvectionBoundaryIntegration() = default;

  virtual void operator()(dealii::MeshWorker::DoFInfo<dim>& dinfo, dealii::MeshWorker::IntegrationInfo<dim>& info) const override;
private:
};

template<unsigned int dim>
class AdvectionFaceIntegration : public AdvectionElementIntegration<dim> {
public:
  AdvectionFaceIntegration(dealii::Vector<double> const& old_solution, double const currTime, double const timeStep);

  virtual ~AdvectionFaceIntegration() = default;

  virtual void operator()(dealii::MeshWorker::DoFInfo<dim>& dinfo1, dealii::MeshWorker::DoFInfo<dim>& dinfo2, dealii::MeshWorker::IntegrationInfo<dim>& info1, dealii::MeshWorker::IntegrationInfo<dim>& info2) const override;
private:
};

#endif
