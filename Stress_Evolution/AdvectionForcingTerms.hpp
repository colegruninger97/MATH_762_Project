#ifndef ADVECTIONFORCINGTERMS_HPP_
#define ADVECTIONFORCINGTERMS_HPP_

#include <deal.II/base/point.h>

/// Implement a forcing term
template<unsigned int dim>
double boundary_condition(dealii::Point<dim> const& p, double const t) {
  if( p(1)>t && p(1)<0.5+t ) {
    return 0.5;
  }
  return 0.0;
  //return p(0) + p(1)*t;
}

/// Implement a forcing term
template<unsigned int dim>
double forcing(dealii::Point<dim> const& p, double const t) {
  //return p(1) + 2.0*t;
  return 0.0;
}

/// Implement the advective velocity
template<unsigned int dim>
dealii::Tensor<1, dim> beta(dealii::Point<dim> const& p, double const t) {
  dealii::Point<dim> wind_field;
  //wind_field(0) = -p(1);
  //wind_field(1) = p(0);
  //wind_field /= wind_field.norm();
  wind_field(0) = 0.0;
  wind_field(1) = 1.0;
  return wind_field;
}

#endif
