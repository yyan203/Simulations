/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Ravi Agrawal (Northwestern U)
------------------------------------------------------------------------- */

#define DEB 0

#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "fix_indent.h"
#include "atom.h"
#include "input.h"
#include "variable.h"
#include "domain.h"
#include "lattice.h"
#include "update.h"
#include "modify.h"
#include "output.h"
#include "respa.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NONE,SPHERE,CYLINDER,PLANE,TRIANGLE};
//enum{NONE,SPHERE,CYLINDER,PLANE};
enum{INSIDE,OUTSIDE};

/* ---------------------------------------------------------------------- */

FixIndent::FixIndent(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal fix indent command");

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;

  k = force->numeric(FLERR,arg[3]);
  k3 = k/3.0;
#if DEB ==1
  printf("k3=%f\n",k3);
#endif
  PI = 4.0*atan(1.0);
  //iaxis = -1.0;

  // read options from end of input line

  options(narg-4,&arg[4]);

  // setup scaling

  if (scaleflag && domain->lattice == NULL)
    error->all(FLERR,"Use of fix indent with undefined lattice");

  double xscale,yscale,zscale;
  if (scaleflag) {
    xscale = domain->lattice->xlattice;
    yscale = domain->lattice->ylattice;
    zscale = domain->lattice->zlattice;
  }
  else xscale = yscale = zscale = 1.0;

  // apply scaling factors to geometry

  //Fenglin add:06/01/12
  if (istyle == SPHERE || istyle == CYLINDER || istyle == TRIANGLE) {
  //Fenglin add:06/01/12
  //original lammps
  //if (istyle == SPHERE || istyle == CYLINDER ) {
  //original lammps
    if (!xstr) xvalue *= xscale;
    if (!ystr) yvalue *= yscale;
    if (!zstr) zvalue *= zscale;
    if (!rstr) rvalue *= xscale;
  } else if (istyle == PLANE) {
    if (cdim == 0 && !pstr) pvalue *= xscale;
    else if (cdim == 1 && !pstr) pvalue *= yscale;
    else if (cdim == 2 && !pstr) pvalue *= zscale;
  } else error->all(FLERR,"Illegal fix indent command");

  varflag = 0;
  if (xstr || ystr || zstr || rstr || pstr) varflag = 1;

  indenter_flag = 0;
  indenter[0] = indenter[1] = indenter[2] = indenter[3] = 0.0;
}

/* ---------------------------------------------------------------------- */

FixIndent::~FixIndent()
{
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;
  delete [] rstr;
  delete [] pstr;
}

/* ---------------------------------------------------------------------- */

int FixIndent::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= THERMO_ENERGY;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixIndent::init()
{
  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0) 
      error->all(FLERR,"Variable name for fix indent does not exist");
    if (!input->variable->equalstyle(xvar))
      error->all(FLERR,"Variable for fix indent is invalid style");
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0) 
      error->all(FLERR,"Variable name for fix indent does not exist");
    if (!input->variable->equalstyle(yvar))
      error->all(FLERR,"Variable for fix indent is not equal style");
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0) 
      error->all(FLERR,"Variable name for fix indent does not exist");
    if (!input->variable->equalstyle(zvar))
      error->all(FLERR,"Variable for fix indent is not equal style");
  }
  if (rstr) {
    rvar = input->variable->find(rstr);
    if (rvar < 0) 
      error->all(FLERR,"Variable name for fix indent does not exist");
    if (!input->variable->equalstyle(rvar))
      error->all(FLERR,"Variable for fix indent is not equal style");
  }
  if (pstr) {
    pvar = input->variable->find(pstr);
    if (pvar < 0) 
      error->all(FLERR,"Variable name for fix indent does not exist");
    if (!input->variable->equalstyle(pvar))
      error->all(FLERR,"Variable for fix indent is not equal style");
  }

  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixIndent::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    post_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }
}

/* ---------------------------------------------------------------------- */

void FixIndent::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixIndent::post_force(int vflag)
{
  // indenter values, 0 = energy, 1-3 = force components
  // wrap variable evaluations with clear/add
  
  if (varflag) modify->clearstep_compute();

  indenter_flag = 0;
  indenter[0] = indenter[1] = indenter[2] = indenter[3] = 0.0;

  // spherical indenter

  if (istyle == SPHERE) {

    // ctr = current indenter center
    // remap into periodic box

    double ctr[3];
    if (xstr) ctr[0] = input->variable->compute_equal(xvar);
    else ctr[0] = xvalue;
    if (ystr) ctr[1] = input->variable->compute_equal(yvar);
    else ctr[1] = yvalue;
    if (zstr) ctr[2] = input->variable->compute_equal(zvar);
    else ctr[2] = zvalue;
    domain->remap(ctr);

    double radius;
    if (rstr) radius = input->variable->compute_equal(rvar);
    else radius = rvalue;

    double **x = atom->x;
    double **f = atom->f;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    double delx,dely,delz,r,dr,fmag,fx,fy,fz;

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
	delx = x[i][0] - ctr[0];
	dely = x[i][1] - ctr[1];
	delz = x[i][2] - ctr[2];
	domain->minimum_image(delx,dely,delz);
	r = sqrt(delx*delx + dely*dely + delz*delz);
	if (side == OUTSIDE) {
	  dr = r - radius;
	  fmag = k*dr*dr;
	} else {
	  dr = radius - r;
	  fmag = -k*dr*dr;
	}
	if (dr >= 0.0) continue;
	fx = delx*fmag/r;
	fy = dely*fmag/r;
	fz = delz*fmag/r;
	f[i][0] += fx;
	f[i][1] += fy;
	f[i][2] += fz;
	indenter[0] -= k3 * dr*dr*dr;
	indenter[1] -= fx;
	indenter[2] -= fy;
	indenter[3] -= fz;
      }

  // cylindrical indenter

  } else if (istyle == CYLINDER) {

    // ctr = current indenter axis
    // remap into periodic box
    // 3rd coord is just near box for remap(), since isn't used
	   
    double ctr[3];
    if (cdim == 0) {
      ctr[0] = domain->boxlo[0];
      if (ystr) ctr[1] = input->variable->compute_equal(yvar);
      else ctr[1] = yvalue;
      if (zstr) ctr[2] = input->variable->compute_equal(zvar);
      else ctr[2] = zvalue;
    } else if (cdim == 1) {
      if (xstr) ctr[0] = input->variable->compute_equal(xvar);
      else ctr[0] = xvalue;
      ctr[1] = domain->boxlo[1];
      if (zstr) ctr[2] = input->variable->compute_equal(zvar);
      else ctr[2] = zvalue;
    } else {
      if (xstr) ctr[0] = input->variable->compute_equal(xvar);
      else ctr[0] = xvalue;
      if (ystr) ctr[1] = input->variable->compute_equal(yvar);
      else ctr[1] = yvalue;
      ctr[2] = domain->boxlo[2];
    }
    domain->remap(ctr);

    double radius;
    if (rstr) radius = input->variable->compute_equal(rvar);
    else radius = rvalue;

    double **x = atom->x;
    double **f = atom->f;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    
    double delx,dely,delz,r,dr,fmag,fx,fy,fz;
    
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
	if (cdim == 0) {
	  delx = 0;
	  dely = x[i][1] - ctr[1];
	  delz = x[i][2] - ctr[2];
	} else if (cdim == 1) {
	  delx = x[i][0] - ctr[0];
	  dely = 0;
	  delz = x[i][2] - ctr[2];
	} else {
	  delx = x[i][0] - ctr[0];
	  dely = x[i][1] - ctr[1];
	  delz = 0;
	}
	domain->minimum_image(delx,dely,delz);
	r = sqrt(delx*delx + dely*dely + delz*delz);
	if (side == OUTSIDE) {
	  dr = r - radius;
	  fmag = k*dr*dr;
	} else {
	  dr = radius - r;
	  fmag = -k*dr*dr;
	}
	if (dr >= 0.0) continue;
	fx = delx*fmag/r;
	fy = dely*fmag/r;
	fz = delz*fmag/r;
	f[i][0] += fx;
	f[i][1] += fy;
	f[i][2] += fz;
	indenter[0] -= k3 * dr*dr*dr;
	indenter[1] -= fx;
	indenter[2] -= fy;
	indenter[3] -= fz;
      }

  // triangular indenter (or v-shape indenter)
  } else if (istyle == TRIANGLE) {

    // ctr = current indenter axis
    // remap into periodic box
    // 3rd coord is just near box for remap(), since isn't used
    double ctr[3];
    if (cdim == 0) {
      ctr[0] = domain->boxlo[0];
      if (ystr) ctr[1] = input->variable->compute_equal(yvar);
      else ctr[1] = yvalue;
      if (zstr) ctr[2] = input->variable->compute_equal(zvar);
      else ctr[2] = zvalue;
    } else if (cdim == 1) {
      if (xstr) ctr[0] = input->variable->compute_equal(xvar);
      else ctr[0] = xvalue;
      ctr[1] = domain->boxlo[1];
      if (zstr) ctr[2] = input->variable->compute_equal(zvar);
      else ctr[2] = zvalue;
    } else {
      if (xstr) ctr[0] = input->variable->compute_equal(xvar);
      else ctr[0] = xvalue;
      if (ystr) ctr[1] = input->variable->compute_equal(yvar);
      else ctr[1] = yvalue;
      ctr[2] = domain->boxlo[2];
    }
    domain->remap(ctr);

    double radius;
    if (rstr) radius = input->variable->compute_equal(rvar);
    else radius = rvalue;

    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    
    double delx,dely,delz,r,dr,fmag,fx,fy,fz;
    double delx1,dely1,delz1,r1,dr1;
    double delx2,dely2,delz2,r2,dr2; // for edge
    double my_c;
    double st,ct;
    st=fabs(sin(atan(tangle))); 
    ct=fabs(cos(atan(tangle))); 
#if DEB ==1
    //printf("cos(tangle)=%f\tsin(tangle)=%f\n",ct,st);
#endif
    /*Fenglin add more variables for tweaked version */
    double rcrt,stheta,ctheta; 
    ////////////////////////////////////////////////////

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
/* Fenglin Tweaked Cylinder-> V-Shape Indenter with round(or i.e. cylindrical) tip */
//1) V-shape indenter with a finite degree edge-edge angle defined by 2*arctan(tangle)
//2) Implement a semi-infinite v-indenter by tweaking the (delx1,dely,delz1) array assuming the indenter is always lowered along the negative axis defined by parameter idim & iaxis
	if (cdim == 0) {
	  delx = 0;
	  dely = x[i][1] - ctr[1];
	  delz = x[i][2] - ctr[2];

          delx1= 0;
          delx2= 0;
          if(idim==1){
          dely1= iaxis*st;
          delz1= delz/fabs(delz)*ct;
          dely2=-1*iaxis*ct;
          delz2= delz/fabs(delz)*st;
          }
          else if(idim==2){
          dely1= dely/fabs(dely)*ct;
          delz1= iaxis*st;
          dely2= dely/fabs(dely)*st;
          delz2=-1*iaxis*ct;
          }
	} else if (cdim == 1) {
	  delx = x[i][0] - ctr[0];
	  dely = 0;
	  delz = x[i][2] - ctr[2];
     
          if(idim==0){
          delx1= delx/fabs(delx)*ct;
          delz1= iaxis*st; 
          delx2= delx/fabs(delx)*st;
          delz2=-1*iaxis*ct; 
          }
          else if(idim==2){
          delx1= iaxis*st;
          delz1= delz/fabs(delz)*ct;  
          delx2=-1*iaxis*ct; 
          delz2= delz/fabs(delz)*st;
          }
          dely1= 0;
          dely2= 0;

	} else {
	// indenter is infinite in z diretion
	  delx = x[i][0] - ctr[0];
	  dely = x[i][1] - ctr[1];
	  delz = 0;
         
          if(idim==0) {
          delx1= iaxis*st;
          dely1= dely/fabs(dely)*ct;
          delx2= -1*iaxis*ct;
          dely2= dely/fabs(dely)*st;
          }
	// indentation direction is along y diretion
          else if(idim==1){
          delx1= delx/fabs(delx)*ct;
          dely1= iaxis*st;
          delx2= delx/fabs(delx)*st;
          dely2= -1.0*iaxis*ct;
          }
          delz1= 0;
          delz2= 0;
	}
	domain->minimum_image(delx,dely,delz);
	r = sqrt(delx*delx + dely*dely + delz*delz);
	r1= sqrt(delx1*delx1 + dely1*dely1 + delz1*delz1);
	r2= sqrt(delx2*delx2 + dely2*dely2 + delz2*delz2);
        
        if(cdim == 0){
           if(idim==1){
           stheta=fabs(delz/r);
           ctheta=iaxis*(dely/r);
           }
           else if(idim==2){
           stheta=fabs(dely/r);
           ctheta=iaxis*(delz/r);
           }
        } else if (cdim == 1){
           if(idim==0){
           stheta=fabs(delz/r);
           ctheta=iaxis*(delx/r);
           }
           else if(idim==2){
           stheta=fabs(delx/r);
           ctheta=iaxis*(delz/r);
           }
        } else if (cdim == 2){
           if(idim==0){
           stheta=fabs(dely/r);
           ctheta=iaxis*(delx/r);
           }
           else if(idim==1){
           stheta=fabs(delx/r);
           ctheta=iaxis*dely/r;
           }
        }
	
       // rcrt=radius/(ctheta+stheta/tangle);
        rcrt=radius/(st*ctheta+stheta*ct);//rcrt=radius/cos(theta-(Pi/2-alpha))
        if(ctheta>1.0||stheta>1.0){printf("warning in fix_indent\n");}
  
        //Use ctheta vs st as criterion to determine whether the atom is close to tip or not
        // If ctheta <=st,the atom is close to edge side
        // If ctehta > st,the atom is close to round tip side 
	if (side == OUTSIDE) {
          if(ctheta<=st){
	    dr = (r - rcrt)*radius/rcrt;
	    //dr = (r - rcrt)*radius*st/rcrt;
	    fmag = k*dr*dr;
          }
          else{
            dr = r - radius;
	    fmag = k*dr*dr;
          }
	} else {
          if(ctheta<=st){
	    dr = (rcrt - r)*radius/rcrt;
	    //dr = (rcrt - r)*radius*st/rcrt;
	    fmag = -k*dr*dr;
          }
          else{
            dr = radius -r;
	    fmag = -k*dr*dr;
          }
	}

	if (dr >= 0.0) continue;
	/*fx = delx*fmag/r;
	  fy = dely*fmag/r;
	  fz = delz*fmag/r;*/
	  //closer to edge
        if(ctheta<=st){
	
	my_c=v[i][0]*st*delx/fabs(delx)+(v[i][1]-indentation_v*idir)*ct; // if my_c>0, atom move towards the indenter, otherwise, move away from it;

	  if(fabs(my_c)<0.00001)
	  {
	  fx = delx1*fmag/r1;
	  fy = dely1*fmag/r1;
	  fz = 0;
	  }
	  else if (my_c>0) //friction force is towards the tip apex
	  {
	  fx = delx1*fmag/r1-delx2*kfrict*fmag/r2;
	  fy = dely1*fmag/r1-ct*kfrict*fmag/r2;
	  fz = 0;
          //printf("delx1 %f delx2 %f fmag %f r2 %f kfrict %f %f %f fx %f fy %f\n",delx1,delx2,fmag,r2,kfrict,delx1*fmag/r1,-delx2*kfrict*fmag/r2,fx,fy);
	  }
	  else //friction force is away from the tip apex
	  {
	  fx = delx1*fmag/r1+delx2*kfrict*fmag/r2;
	  fy = dely1*fmag/r1+dely2*kfrict*fmag/r2;
	  fz = 0;
	  }

        }
	//closer to sphere cap
        else{
	if(fabs(delx)<0.00001) my_c=v[i][0];
	else my_c=v[i][0]*delx/fabs(delx)*fabs(dely)/r + (v[i][1]-indentation_v*idir)*fabs(delx)/r;

	  if(fabs(my_c)<0.00001)
	  {
          fx = delx*fmag/r;
          fy = dely*fmag/r;
          fz = 0;
	  }
	  else if (my_c>0) //friction force is towards the tip apex
	  {
          fx = delx*fmag/r-delx/fabs(delx)*fabs(dely)*kfrict*fmag/r;
          fy = dely*fmag/r-fabs(delx)*kfrict*fmag/r;
          fz = 0;
          //printf("delx %f dely %f fmag %f r %f kfrict %f %f %f fx %f fy %f\n",delx,dely,fmag,r, kfrict,delx*fmag/r,-delx/fabs(delx)*fabs(dely)*kfrict*fmag/r,fx,fy);
	  }
	  else  //friction force is away from the tip apex
	  {
          fx = delx*fmag/r+delx/fabs(delx)*fabs(dely)*kfrict*fmag/r;
          fy = dely*fmag/r+fabs(delx)*kfrict*fmag/r;
          fz = 0;
	  }
        }
	f[i][0] += fx;
	f[i][1] += fy;
	f[i][2] += fz;
	indenter[0] -= k3 * dr*dr*dr;
	indenter[1] -= fx;
	indenter[2] -= fy;
	indenter[3] -= fz;
      }

  // planar indenter
  } else {

    // plane = current plane position
	      
    double plane;
    if (pstr) plane = input->variable->compute_equal(pvar);
    else plane = pvalue;
    
    double **x = atom->x;
    double **v = atom->v;//Yongjian
    double **f = atom->f;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    
    double my_c;//compute relative speed, assuming indenter is moving toward left or right (in x direction)
    double dr,fatom,fratom;
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
	dr = planeside * (plane - x[i][cdim]);
	if (dr >= 0.0) continue;
	fatom = -planeside * k*dr*dr;
	// calculate relative velocity of atom to plane slider
	// j
	my_c = v[i][0]-50000/540;
	if(fabs(my_c)<0.00001) // Yongjian
	{ //Yongjian
	f[i][cdim] += fatom;
	indenter[0] -= k3 * dr*dr*dr;
	indenter[cdim+1] -= fatom;

	} else if (my_c>0)  // 
	{
	f[i][cdim] += fatom;
	fratom = fabs(fatom*kfrict); // only test x direction

	//printf("my_c=>%f kfrict=>%f ",my_c,kfrict); 
	//printf("fratom=>%f\n",fratom); 
	//printf("%dth cdim:%d,before_frict:%e,",i,cdim,f[i][0]); 
	//printf("b:%e ",f[i][0]); 
	f[i][0] -= fratom;
	//printf("after_frict:%e  ",f[i][0]); 
	//printf("a:%e",f[i][0]); 
	//printf("\n");
	//if(i==311) printf("311th N:%f,F:%f\n",fatom,fratom); 
	//printf("%dth N:%f,F:%f\n",i,fatom,fratom); 

	indenter[0] -= k3 * dr*dr*dr;
	indenter[cdim+1] -= fatom;
	indenter[1] += fratom ;
	}
	else
	{
	f[i][cdim] += fatom;
	fratom = fabs(fatom*kfrict); // only test x direction
	//printf("my_c=>%f kfrict=>%f ",my_c,kfrict); 
	//printf("%dth N:%f,F:%f\n",i,fatom,fratom); 
	//printf("%dth cdim:%d,before_frict:%e,",i,cdim,f[i][0]); 
	f[i][0] += fratom;
	//printf("after_frict:%e  ",f[i][0]); 
	//printf("%dth N:%f,F:%f\n",i,fatom,fratom); 
	indenter[0] -= k3 * dr*dr*dr;
	indenter[cdim+1] -= fatom;
	indenter[1] -= fratom ;
	}
      }
      //printf("indenter force: %f %f %f\n",indenter[1],indenter[2],indenter[3]);
  }

  if (varflag) modify->addstep_compute(update->ntimestep + 1);
}

/* ---------------------------------------------------------------------- */

void FixIndent::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixIndent::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   energy of indenter interaction
------------------------------------------------------------------------- */

double FixIndent::compute_scalar()
{
  // only sum across procs one time

  if (indenter_flag == 0) {
    MPI_Allreduce(indenter,indenter_all,4,MPI_DOUBLE,MPI_SUM,world);
    indenter_flag = 1;
  }
  return indenter_all[0];
}

/* ----------------------------------------------------------------------
   components of force on indenter
------------------------------------------------------------------------- */

double FixIndent::compute_vector(int n)
{
  // only sum across procs one time

  if (indenter_flag == 0) {
    MPI_Allreduce(indenter,indenter_all,4,MPI_DOUBLE,MPI_SUM,world);
    indenter_flag = 1;
  }
  return indenter_all[n+1];
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line 
------------------------------------------------------------------------- */

void FixIndent::options(int narg, char **arg)
{
  if (narg < 0) error->all(FLERR,"Illegal fix indent command");

  istyle = NONE;
  xstr = ystr = zstr = rstr = pstr = NULL;
  xvalue = yvalue = zvalue = rvalue = pvalue = 0.0;
  scaleflag = 1;
  side = OUTSIDE;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"sphere") == 0) {
      if (iarg+5 > narg) error->all(FLERR,"Illegal fix indent command");

      if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
	int n = strlen(&arg[iarg+1][2]) + 1;
	xstr = new char[n];
	strcpy(xstr,&arg[iarg+1][2]);
      } else xvalue = force->numeric(FLERR,arg[iarg+1]);
      if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
	int n = strlen(&arg[iarg+2][2]) + 1;
	ystr = new char[n];
	strcpy(ystr,&arg[iarg+2][2]);
      } else yvalue = force->numeric(FLERR,arg[iarg+2]);
      if (strstr(arg[iarg+3],"v_") == arg[iarg+3]) {
	int n = strlen(&arg[iarg+3][2]) + 1;
	zstr = new char[n];
	strcpy(zstr,&arg[iarg+3][2]);
      } else zvalue = force->numeric(FLERR,arg[iarg+3]);
      if (strstr(arg[iarg+4],"v_") == arg[iarg+4]) {
	int n = strlen(&arg[iarg+4][2]) + 1;
	rstr = new char[n];
	strcpy(rstr,&arg[iarg+4][2]);
      } else rvalue = force->numeric(FLERR,arg[iarg+4]);

      istyle = SPHERE;
      iarg += 5;

    } else if (strcmp(arg[iarg],"cylinder") == 0) {
      if (iarg+5 > narg) error->all(FLERR,"Illegal fix indent command");

      if (strcmp(arg[iarg+1],"x") == 0) {
	cdim = 0;
	if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
	  int n = strlen(&arg[iarg+2][2]) + 1;
	  ystr = new char[n];
	  strcpy(ystr,&arg[iarg+2][2]);
	} else yvalue = force->numeric(FLERR,arg[iarg+2]);
	if (strstr(arg[iarg+3],"v_") == arg[iarg+3]) {
	  int n = strlen(&arg[iarg+3][2]) + 1;
	  zstr = new char[n];
	  strcpy(zstr,&arg[iarg+3][2]);
	} else zvalue = force->numeric(FLERR,arg[iarg+3]);
      } else if (strcmp(arg[iarg+1],"y") == 0) {
	cdim = 1;
	if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
	  int n = strlen(&arg[iarg+2][2]) + 1;
	  xstr = new char[n];
	  strcpy(xstr,&arg[iarg+2][2]);
	} else xvalue = force->numeric(FLERR,arg[iarg+2]);
	if (strstr(arg[iarg+3],"v_") == arg[iarg+3]) {
	  int n = strlen(&arg[iarg+3][2]) + 1;
	  zstr = new char[n];
	  strcpy(zstr,&arg[iarg+3][2]);
	} else zvalue = force->numeric(FLERR,arg[iarg+3]);
      } else if (strcmp(arg[iarg+1],"z") == 0) {
	cdim = 2;
	if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
	  int n = strlen(&arg[iarg+2][2]) + 1;
	  xstr = new char[n];
	  strcpy(xstr,&arg[iarg+2][2]);
	} else xvalue = force->numeric(FLERR,arg[iarg+2]);
	if (strstr(arg[iarg+3],"v_") == arg[iarg+3]) {
	  int n = strlen(&arg[iarg+3][2]) + 1;
	  ystr = new char[n];
	  strcpy(ystr,&arg[iarg+3][2]);
	} else yvalue = force->numeric(FLERR,arg[iarg+3]);
      } else error->all(FLERR,"Illegal fix indent command");

      if (strstr(arg[iarg+4],"v_") == arg[iarg+4]) {
	int n = strlen(&arg[iarg+4][2]) + 1;
	rstr = new char[n];
	strcpy(rstr,&arg[iarg+4][2]);
      } else rvalue = force->numeric(FLERR,arg[iarg+4]);

      istyle = CYLINDER;
      iarg += 5;

    } else if (strcmp(arg[iarg],"triangle") == 0) {
#if DEB == 1
      printf("entering triangle style read-in subroutine\n");
#endif
      // Implementation-1: use angle style to specific Edge_angle of trinagle indenter
      // fix fix_ID region indent K(force_constant) triangle central_axis(x/y/z) coordinate-1 coordinate-2 Radius angle Edge_Angle 
      if (iarg+5 > narg) error->all(FLERR,"Illegal fix indent command");

      if (strcmp(arg[iarg+1],"x") == 0) {
	cdim = 0;
	if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
	  int n = strlen(&arg[iarg+2][2]) + 1;
	  ystr = new char[n];
	  strcpy(ystr,&arg[iarg+2][2]);
	} else yvalue = force->numeric(FLERR,arg[iarg+2]);
	if (strstr(arg[iarg+3],"v_") == arg[iarg+3]) {
	  int n = strlen(&arg[iarg+3][2]) + 1;
	  zstr = new char[n];
	  strcpy(zstr,&arg[iarg+3][2]);
	} else zvalue = force->numeric(FLERR,arg[iarg+3]);
      } else if (strcmp(arg[iarg+1],"y") == 0) {
	cdim = 1;
	if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
	  int n = strlen(&arg[iarg+2][2]) + 1;
	  xstr = new char[n];
	  strcpy(xstr,&arg[iarg+2][2]);
	} else xvalue = force->numeric(FLERR,arg[iarg+2]);
	if (strstr(arg[iarg+3],"v_") == arg[iarg+3]) {
	  int n = strlen(&arg[iarg+3][2]) + 1;
	  zstr = new char[n];
	  strcpy(zstr,&arg[iarg+3][2]);
	} else zvalue = force->numeric(FLERR,arg[iarg+3]);
      } else if (strcmp(arg[iarg+1],"z") == 0) {
	cdim = 2;
	if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
	  int n = strlen(&arg[iarg+2][2]) + 1;
	  xstr = new char[n];
	  strcpy(xstr,&arg[iarg+2][2]);
	} else xvalue = force->numeric(FLERR,arg[iarg+2]);
	if (strstr(arg[iarg+3],"v_") == arg[iarg+3]) {
	  int n = strlen(&arg[iarg+3][2]) + 1;
	  ystr = new char[n];
	  strcpy(ystr,&arg[iarg+3][2]);
	} else yvalue = force->numeric(FLERR,arg[iarg+3]);
      } else error->all(FLERR,"Illegal fix indent command");
#if DEB == 1
      printf("finish read-in axis of indenter\n");
#endif

      if (strstr(arg[iarg+4],"v_") == arg[iarg+4]) {
	int n = strlen(&arg[iarg+4][2]) + 1;
	rstr = new char[n];
	strcpy(rstr,&arg[iarg+4][2]);
      } else rvalue = force->numeric(FLERR,arg[iarg+4]);

/*      if (strstr(arg[iarg+5],"v_") == arg[iarg+5]) {
	int n = strlen(&arg[iarg+5][2]) + 1;
	rstr = new char[n];
	strcpy(rstr,&arg[iarg+5][2]);
      } else rvalue = force->numeric(FLERR,arg[iarg+5]);
*/
      istyle = TRIANGLE;
      iarg += 5;

    } else if (strcmp(arg[iarg],"plane") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix indent command");
      if (strcmp(arg[iarg+1],"x") == 0) cdim = 0;
      else if (strcmp(arg[iarg+1],"y") == 0) cdim = 1;
      else if (strcmp(arg[iarg+1],"z") == 0) cdim = 2;
      else error->all(FLERR,"Illegal fix indent command");

      if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
	int n = strlen(&arg[iarg+2][2]) + 1;
	pstr = new char[n];
	strcpy(pstr,&arg[iarg+2][2]);
      } else pvalue = force->numeric(FLERR,arg[iarg+2]);

      if (strcmp(arg[iarg+3],"lo") == 0) planeside = -1;
      else if (strcmp(arg[iarg+3],"hi") == 0) planeside = 1;
      else error->all(FLERR,"Illegal fix indent command");
      istyle = PLANE;
      iarg += 4;

    } else if (strcmp(arg[iarg],"units") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix indent command");
      if (strcmp(arg[iarg+1],"box") == 0) scaleflag = 0;
      else if (strcmp(arg[iarg+1],"lattice") == 0) scaleflag = 1;
      else error->all(FLERR,"Illegal fix indent command");
      iarg += 2;

    } else if (strcmp(arg[iarg],"side") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix indent command");
      if (strcmp(arg[iarg+1],"in") == 0) side = INSIDE;
      else if (strcmp(arg[iarg+1],"out") == 0) side = OUTSIDE;
      else error->all(FLERR,"Illegal fix indent command");
      iarg += 2;
    /* Fenglin add: new argument "angle" to specific edge angle in degree format for TRIANGLE style */
    /* Current argument "angle" is only compatible with TRIANGLE style */
    } else if (strcmp(arg[iarg],"angle") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix indent command");
      tangle=tan(force->numeric(FLERR,arg[iarg+1])*PI/180/2);             
#if DEB ==1
      printf("tangle=%f\n",tangle);
#endif
      iarg += 2;
    } else if (strcmp(arg[iarg],"orient") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix indent command");
      if(strcmp(arg[iarg+1],"x")==0) idim=0;
      else if(strcmp(arg[iarg+1],"y")==0) idim=1;
      else if(strcmp(arg[iarg+1],"z")==0) idim=2;
      if(strcmp(arg[iarg+2],"plus")==0) iaxis=1;
      else if(strcmp(arg[iarg+2],"minus")==0) iaxis=-1;
      else error->all(FLERR,"Illegal fix indent command");
      iarg += 3;
    /* Yongjian add: new argument "frict" to account for frictional force in TRIANGLE style */  
    /* only consider that the frictional coefficient is constant */
    /* the direction and value of kinetic friction force depends on the relative velocity of indenter and the contact atoms*/
    } else if (strcmp(arg[iarg],"frict") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix indent command");
      kfrict=force->numeric(FLERR,arg[iarg+1]);
      if(strcmp(arg[iarg+2],"approach")==0) idir=-1;
      else if(strcmp(arg[iarg+2],"retract")==0) idir=1;
      else error->all(FLERR,"Illegal fix indent command");
      indentation_v=force->numeric(FLERR,arg[iarg+3]);
      iarg += 4;
    } 
    else error->all(FLERR,"Illegal fix indent command");
  }
}
