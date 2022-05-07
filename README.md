# MATH_762_Project

This repository contains two separate finite element solvers which were implemented using the open source finite element library dealii : https://www.dealii.org/.
As its name suggests, the folder Stokes_Solver contains a Taylor-Hood finite element implementation which solves the 2-D Stokes equations on the unit square with Dirichlet boundary conditions.

The folder Stress_Evolution contains a Discontinuous Galerkin finite element implementation which solves the linear advection equation 

<center> &tau;<sub>t = <strong>u</strong> &middot;&nabla;&tau;</center>   
    <br/><br/>
in two spaital dimensions on the unit square. Currently, the inflow boundary conditions are prescribed.
    
In the future, my hope is to clean up the DG linear advection solver, verify that it's convergent, and, of course, bug free.
  
  

