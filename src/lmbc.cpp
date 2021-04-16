/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//                              LICENSE NOTICES                                //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////

//  This file is part of the program
//  Levenberg-Marquardt C++ non-linear minimization algorithm (henceforth LM++)
//  Copyright (C) 2021  Manuel Portilheiro
//
//  LM++ is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 3 of the License, or
//  (at your option) any later version.
//
//  LM++ is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with LM++.  If not, see <https://www.gnu.org/licenses/>.
//
//	LM++ is a modification of the Levenberg - Marquardt non-linear minimization
//  algorithm, Copyright (C) 2004  Manolis Lourakis (lourakis at ics forth gr),
//  version 2.6. A copy of the original copyright notice follows.

//  ORIGINAL COPYRIGHT NOTICE:

	/////////////////////////////////////////////////////////////////////////////////
	// 
	//  Levenberg - Marquardt non-linear minimization algorithm
	//  Copyright (C) 2004-05  Manolis Lourakis (lourakis at ics forth gr)
	//  Institute of Computer Science, Foundation for Research & Technology - Hellas
	//  Heraklion, Crete, Greece.
	//
	//  This program is free software; you can redistribute it and/or modify
	//  it under the terms of the GNU General Public License as published by
	//  the Free Software Foundation; either version 2 of the License, or
	//  (at your option) any later version.
	//
	//  This program is distributed in the hope that it will be useful,
	//  but WITHOUT ANY WARRANTY; without even the implied warranty of
	//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	//  GNU General Public License for more details.
	//
	/////////////////////////////////////////////////////////////////////////////////

/******************************************************************************** 
 * Box-constrained Levenberg-Marquardt nonlinear minimization.
 ********************************************************************************/

//#include <cstdio>
//#include <cstdlib>
#include <cmath>
//#include <cfloat>
#include <limits>
#include <vector>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include "levmar.h"
#include "compiler_lmpp.h"
#include "misc.h"
#include "external_wrappers.h"
#include "Axb.h"

//#define EPSILON       1E-12
//#define ONE_THIRD     0.3333333334 /* 1.0/3.0 */
#define LPMM_LSITMAX    150 /* max #iterations for line search */
#define LMPP_SING_EPS   FLOATTYPE(1e-24)

//#define _POW_         2.1


/* find the median of 3 numbers */
template<typename FLOATTYPE>
static inline FLOATTYPE median3( FLOATTYPE a, FLOATTYPE b, FLOATTYPE c )
{
	return a >= b ? ( c >= a ? a : ( c <= b ? b : c ) )
				  : ( c >= b ? b : ( c <= a ? a : c ) );
}


/* Projections to feasible set \Omega: P_{\Omega}(y) := arg min { ||x - y|| : x \in \Omega},  y \in R^m */

/* project vector p to a box shaped feasible set. p is a mx1 vector.
 * Either lb, ub can be NULL. If not NULL, they are mx1 vectors
 */
template<typename FLOATTYPE>
static void boxProject( FLOATTYPE* p, FLOATTYPE* lb, FLOATTYPE* ub, int m )
{
	if( !lb )     /* no lower bounds   */
	{
		if( !ub ) /* no upper bounds   */
			return;
		else      /* upper bounds only */
			for( int i = m-1; i >= 0; --i )
				if( p[i] > ub[i] ) p[i] = ub[i];
	}
	else
	{
		if( !ub ) /* lower bounds only */
			for( int i = m-1; i >= 0; --i )
				if( p[i] < lb[i] ) p[i] = lb[i];
		else      /* box bounds */
			for( int i = m-1; i >= 0; --i )
				p[i] = median3( lb[i], p[i], ub[i] );
	}
}

/* pointwise scaling of bounds with the mx1 vector scl. If div=1 scaling is by 1./scl.
 * Either lb, ub can be NULL. If not NULL, they are mx1 vectors
 */
template<typename FLOATTYPE>
static void boxScale( FLOATTYPE* lb, FLOATTYPE* ub, FLOATTYPE* scl, int m, int div )
{
	if( !lb )     /* no lower bounds   */
	{
		if( !ub ) /* no upper bounds   */
			return;
		else      /* upper bounds only */
			if( div )
				for( int i = m-1; i >= 0; --i )
					if( ub[i] != std::numeric_limits<FLOATTYPE>::max() ) ub[i] = ub[i] / scl[i];
			else
				for( int i = m-1; i >= 0; --i )
					if( ub[i] != std::numeric_limits<FLOATTYPE>::max() ) ub[i] = ub[i] * scl[i];
	}
	else
	{
		if( !ub ) /* lower bounds only */
			if( div )
				for( int i = m-1; i >= 0; --i )
					if( lb[i] != std::numeric_limits<FLOATTYPE>::min() ) lb[i] = lb[i] / scl[i];
			else
				for( int i = m-1; i >= 0; --i )
					if( lb[i] != std::numeric_limits<FLOATTYPE>::min() ) lb[i] = lb[i] * scl[i];
		else      /* box bounds        */
			if( div )
				for( int i = m-1; i >= 0; --i )
				{
					if( ub[i] != std::numeric_limits<FLOATTYPE>::max() ) ub[i] = ub[i] / scl[i];
					if( lb[i] != std::numeric_limits<FLOATTYPE>::min() ) lb[i] = lb[i] / scl[i];
				}
			else
				for( int i = m-1; i >= 0; --i )
				{
					if( ub[i] != std::numeric_limits<FLOATTYPE>::max() ) ub[i] = ub[i] * scl[i];
					if( lb[i] != std::numeric_limits<FLOATTYPE>::min() ) lb[i] = lb[i] * scl[i];
				}
	}
}

///* compute the norm of a vector in a manner that avoids overflows
// */
//template<typename FLOATTYPE>
//static FLOATTYPE vecnorm( FLOATTYPE* x, int n )
//{
//#ifdef LMPP_HAVE_LAPACK
//#define NRM2 LM_MK_BLAS_NAME(nrm2)
//	extern LM_REAL NRM2(int *n, LM_REAL *dx, int *incx);
//	int one=1;
//
//	return NRM2(&n, x, &one);
//#undef NRM2
//#else // no LAPACK, use the simple method described by Blue in TOMS78
//	register int i;
//	LM_REAL max, sum, tmp;
//
//	for(i=n, max=0.0; i-->0; )
//		if(x[i]>max) max=x[i];
//		else if(x[i]<-max) max=-x[i];
//
//	for(i=n, sum=0.0; i-->0; ){
//		tmp=x[i]/max;
//		sum+=tmp*tmp;
//	}
//
//	return max*(LM_REAL)sqrt(sum);
//#endif /* LMPP_HAVE_LAPACK */
//}

template<typename FLOATTYPE>
struct func_state{
	int        n;
	int*       nfev;
	FLOATTYPE* hx;
	FLOATTYPE* x;
	FLOATTYPE* lb;
	FLOATTYPE* ub;
	void*      adata;
};

template<typename FLOATTYPE>
static void
lnsrch( int m, FLOATTYPE* x, FLOATTYPE f, FLOATTYPE* g, FLOATTYPE* p, FLOATTYPE alpha, FLOATTYPE* xpls,
		FLOATTYPE* ffpls, FPTR<FLOATTYPE> func, func_state<FLOATTYPE>* state,
		int* mxtake, int* iretcd, FLOATTYPE stepmx, FLOATTYPE steptl, FLOATTYPE* sx )
{
	/* Find a next newton iterate by backtracking line search.
	 * Specifically, finds a \lambda such that for a fixed alpha<0.5 (usually 1e-4),
	 * f(x + \lambda*p) <= f(x) + alpha * \lambda * g^T*p
	 *
	 * Translated (with a few changes) from Schnabel, Koontz & Weiss uncmin.f,  v1.3
	 * Main changes include the addition of box projection and modification of the scaling 
	 * logic since uncmin.f operates in the original (unscaled) variable space.
	 
	 * PARAMETERS :
	 
	 *	m       --> dimension of problem (i.e. number of variables)
	 *	x(m)    --> old iterate:	x[k-1]
	 *	f       --> function value at old iterate, f(x)
	 *	g(m)    --> gradient at old iterate, g(x), or approximate
	 *	p(m)    --> non-zero newton step
	 *	alpha   --> fixed constant < 0.5 for line search (see above)
	 *	xpls(m) <--	 new iterate x[k]
	 *	ffpls   <--	 function value at new iterate, f(xpls)
	 *	func    --> name of subroutine to evaluate function
	 *	state   <--> information other than x and m that func requires.
	 *			    state is not modified in xlnsrch (but can be modified by func).
	 *	iretcd  <--	 return code
	 *	mxtake  <--	 boolean flag indicating step of maximum length used
	 *	stepmx  --> maximum allowable step size
	 *	steptl  --> relative step size at which successive iterates
	 *			    considered close enough to terminate algorithm
	 *	sx(m)	  --> diagonal scaling matrix for x, can be NULL
	 
	 *	internal variables
	 
	 *	sln		 newton length
	 *	rln		 relative length of newton step
	 */


	f      *= FLOATTYPE(0.5);
	*mxtake = 0;
	*iretcd = 2;

	FLOATTYPE aux = 0.;
	for( int i = m-1; i >= 0; --i  )
		aux += sq(p[i]);

	FLOATTYPE sln = std::sqrt(aux);
	if( sln > stepmx )
	{
		/*	newton step longer than maximum allowed */
		FLOATTYPE scl = stepmx / sln;
		for( int i = m-1; i >= 0; --i  ) /* p * scl */
			p[i] *= scl;
		sln = stepmx;
	}

	FLOATTYPE slp = zero<FLOATTYPE>();
	FLOATTYPE rln = zero<FLOATTYPE>();
	for( int i = m-1; i >= 0; --i  )
	{
		slp += g[i] * p[i]; /* g^T * p */
		rln  = std::max( rln, std::abs(p[i]) / std::max( std::abs(x[i]), one<FLOATTYPE>() ) );
	}

	FLOATTYPE fpls;
	FLOATTYPE rmnlmb    = steptl / rln;
	FLOATTYPE lambda    = one<FLOATTYPE>();
	FLOATTYPE pfpls     = zero<FLOATTYPE>();
	FLOATTYPE plmbda    = zero<FLOATTYPE>();
	int       firstback = 1;

	/*	check if new iterate satisfactory.  generate new lambda if necessary. */
	for( int j = LPMM_LSITMAX-1; j >= 0; --j  )
	{
		for( int i = m-1; i >= 0; --i )
		{
			xpls[i] = x[i] + lambda * p[i];
		}
		boxProject( xpls, state->lb, state->ub, m ); /* project to feasible set */

		/* evaluate function at new point */
		if( !sx )
		{
			(*func)( xpls, state->hx, m, state->n, state->adata ); ++(*(state->nfev) );
		}
		else
		{
			for( int i = m-1; i >= 0; --i )
				xpls[i] *= sx[i];
			(*func)( xpls, state->hx, m, state->n, state->adata ); ++(*(state->nfev) );
			for( int i = m-1; i >= 0; --i )
				xpls[i] /= sx[i];
		}
		/* ### state->hx = state->x - state->hx, aux = ||state->hx|| */
	#if 1
		aux = levmar_L2nrmxmy( state->hx, state->x, state->hx, state->n );
	#else
		aux = zero<FLOATTYPE>();
		for( int i = 0; i < state->n; ++i )
		{
			state->hx[i] = state->x[i] - state->hx[i];
			aux += sq( state->hx[i] );
		}
	#endif
		fpls = FLOATTYPE(0.5) * aux; *ffpls = aux;

		if( fpls <= f + slp * alpha * lambda ) /* solution found */
		{
			*iretcd = 0;
			if( lambda == one<FLOATTYPE>() && sln > stepmx * FLOATTYPE(.99) )
				*mxtake = 1;
			return;
		}

		/* else : solution not (yet) found */

		/* First find a point with a finite value */
		if( lambda < rmnlmb )
		{   /* no satisfactory xpls found sufficiently distinct from x */
			*iretcd = 1;
			return;
		}
		else
		{   /* calculate new lambda */
			/* modifications to cover non-finite values */
			if( !LMPP_FINITE(fpls) )
			{
				lambda   *= FLOATTYPE(0.1);
				firstback = 1;
			}
			else
			{
				FLOATTYPE tlmbda;
				if( firstback ) /* first backtrack: quadratic fit */
				{
					tlmbda    = -lambda * slp / ( (fpls - f - slp) * two<FLOATTYPE>() );
					firstback = 0;
				}
				else /* all subsequent backtracks: cubic fit */
				{
					FLOATTYPE t1   = fpls - f - lambda * slp;
					FLOATTYPE t2   = pfpls - f - plmbda * slp;
					FLOATTYPE t3   = one<FLOATTYPE>() / (lambda - plmbda);
					FLOATTYPE a3   = FLOATTYPE (3.) * t3 * (t1 / sq(lambda) - t2 / sq(plmbda));
					FLOATTYPE b    = t3 * (t2 * lambda / sq(plmbda) - t1 * plmbda / sq(lambda));
					FLOATTYPE disc = sq(b) - a3 * slp;
					if( disc > sq(b) )
						/* only one positive critical point, must be minimum */
						tlmbda = (-b + ( a3 < 0 ? -std::sqrt(disc) : std::sqrt(disc) )) / a3;
					else
						/* both critical points positive, first is minimum */
						tlmbda = (-b + ( a3 < 0 ? std::sqrt(disc) : -std::sqrt(disc) )) / a3;

					if( tlmbda > lambda * FLOATTYPE(.5) )
						tlmbda = lambda * FLOATTYPE(.5);
				}
				plmbda = lambda;
				pfpls  = fpls;
				if( tlmbda < lambda * FLOATTYPE(.1) )
					lambda *= FLOATTYPE(.1);
				else
					lambda = tlmbda;
			}
		}
	}
	/* this point is reached when the iterations limit is exceeded */
	*iretcd = 1; /* failed */

} /* lnsrch */

 /* 
  * This function seeks the parameter vector p that best describes the measurements
  * vector x under box constraints.
  * More precisely, given a vector function  func : R^m --> R^n with n>=m,
  * it finds p s.t. func(p) ~= x, i.e. the squared second order (i.e. L2) norm of
  * e=x-func(p) is minimized under the constraints lb[i]<=p[i]<=ub[i].
  * If no lower bound constraint applies for p[i], use -DBL_MAX/-FLT_MAX for lb[i];
  * If no upper bound constraint applies for p[i], use DBL_MAX/FLT_MAX for ub[i].
  *
  * This function requires an analytic Jacobian. In case the latter is unavailable,
  * use LEVMAR_BC_DIF() bellow
  *
  * Returns the number of iterations (>=0) if successful, LM_ERROR if failed
  *
  * For details, see C. Kanzow, N. Yamashita and M. Fukushima: "Levenberg-Marquardt
  * methods for constrained nonlinear equations with strong local convergence properties",
  * Journal of Computational and Applied Mathematics 172, 2004, pp. 375-397.
  * Also, see K. Madsen, H.B. Nielsen and O. Tingleff's lecture notes on 
  * unconstrained Levenberg-Marquardt at http://www.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf
  *
  * The algorithm implemented by this function employs projected gradient steps. Since steepest descent
  * is very sensitive to poor scaling, diagonal scaling has been implemented through the dscl argument:
  * Instead of minimizing f(p) for p, f(D*q) is minimized for q=D^-1*p, D being a diagonal scaling
  * matrix whose diagonal equals dscl (see Nocedal-Wright p.27). dscl should contain "typical" magnitudes 
  * for the parameters p. A NULL value for dscl implies no scaling. i.e. D=I.
  * To account for scaling, the code divides the starting point and box bounds pointwise by dscl. Moreover,
  * before calling func and jacf the scaling has to be undone (by multiplying), as should be done with
  * the final point. Note also that jac_q=jac_p*D, where jac_q, jac_p are the jacobians w.r.t. q & p, resp.
  */
template<typename FLOATTYPE>
int levmar_bc_der_impl( FPTR<FLOATTYPE> func,    /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
						FPTR<FLOATTYPE> jacf,    /* function to evaluate the Jacobian \part x / \part p */ 
						FLOATTYPE*      p,       /* I/O: initial parameter estimates. On output has the estimated solution */
						FLOATTYPE*      x,       /* I: measurement vector. NULL implies a zero vector */
						int             m,       /* I: parameter vector dimension (i.e. #unknowns) */
						int             n,       /* I: measurement vector dimension */
						FLOATTYPE*      lb,      /* I: vector of lower bounds. If NULL, no lower bounds apply */
						FLOATTYPE*      ub,      /* I: vector of upper bounds. If NULL, no upper bounds apply */
						FLOATTYPE*      dscl,    /* I: diagonal scaling constants. NULL implies no scaling */
						int             itmax,   /* I: maximum number of iterations */
						FLOATTYPE       opts[4], /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
						                          * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used.
						                          * Note that ||J^T e||_inf is computed on free (not equal to lb[i] or ub[i]) variables only.
						                          */
						FLOATTYPE       info[LMPP_INFO_SZ],
						                         /* O: information regarding the minimization. Set to NULL if don't care
						                          * info[0]= ||e||_2 at initial p.
						                          * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
						                          * info[5]= # iterations,
						                          * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
						                          *                                 2 - stopped by small Dp
						                          *                                 3 - stopped by itmax
						                          *                                 4 - singular matrix. Restart from current p with increased mu 
						                          *                                 5 - no further error reduction is possible. Restart with increased mu
						                          *                                 6 - stopped by small ||e||_2
						                          *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
						                          * info[7]= # function evaluations
						                          * info[8]= # Jacobian evaluations
						                          * info[9]= # linear systems solved, i.e. # attempts for reducing error
						                          */
						FLOATTYPE*      work,    /* working memory at least LM_BC_DER_WORKSZ() reals large, allocated if NULL */
						FLOATTYPE*      covar,   /* O: Covariance matrix corresponding to LS solution; mxm. Set to NULL if not needed. */
						void*           adata )  /* pointer to possibly additional data, passed uninterpreted to func & jacf.
						                          * Set to NULL if not needed
						                          */
{
	if( n < m )
	{
		std::cerr << "levmar_bc_der(): cannot solve a problem with fewer measurements [" << n << "] than unknowns [" << m << "]\n";
		return LMPP_ERROR;
	}

	if( !jacf )
	{
		std::cerr << "No function specified for computing the Jacobian in levmar_bc_der()\n";
		std::cerr << "If no such function is available, use levmar_bc_dif() rather than levmar_bc_der()\n";
		return LMPP_ERROR;
	}

	if( !levmar_box_check(lb, ub, m) )
	{
		std::cerr << "levmar_bc_der(): at least one lower bound exceeds the upper one\n";
		return LMPP_ERROR;
	}

	std::vector<FLOATTYPE> sp_pDp; /* dscl*p or dscl*pDp, (m,1) */
	if( dscl ) /* check that scaling consts are valid */
	{
		for( int i = m-1; i >= 0; --i )
		{
			if( dscl[i] <= zero<FLOATTYPE>() )
			{
				std::cerr << "levmar_bc_der(): scaling constants should be positive (scale " << i << ": " << dscl[i] << " <= 0)\n";
				return LMPP_ERROR;
			}
		}
		sp_pDp.resize(m);
	}

	const int nm = n * m;

	const FLOATTYPE tau     = opts ? opts[0] : FLOATTYPE(LMPP_INIT_MU);
	const FLOATTYPE eps1    = opts ? opts[1] : FLOATTYPE(LMPP_STOP_THRESH);
	const FLOATTYPE eps2    = opts ? opts[2] : FLOATTYPE(LMPP_STOP_THRESH);
	const FLOATTYPE eps3    = opts ? opts[3] : FLOATTYPE(LMPP_STOP_THRESH);
	const FLOATTYPE eps2_sq = sq(eps2);

	std::unique_ptr<FLOATTYPE[]> buf;
	if( !work )
	{
		buf = std::make_unique<FLOATTYPE[]>( LMPP_DER_WORKSZ(m, n) ); // worksz = 2*n+4*m + n*m + m*m;
		work = buf.get();
	}

	/* set up temp work arrays */
	FLOATTYPE* e            = work;               /*                    (n,1) */
	FLOATTYPE* hx           = e            + n;   /* \hat{x}_i,         (n,1) */
	FLOATTYPE* jacTe        = hx           + n;	  /* J^T e_i            (m,1) */
	FLOATTYPE* jac          = jacTe        + m;	  /*                    (n,m) */
	FLOATTYPE* jacTjac      = jac          + nm;  /*                    (m,m) */
	FLOATTYPE* Dp           = jacTjac      + m*m; /*                    (m,1) */
	FLOATTYPE* diag_jacTjac = Dp           + m;   /* diagonal of J^T J, (m,1) */
	FLOATTYPE* pDp          = diag_jacTjac + m;   /* p + Dp,            (m,1) */

	FLOATTYPE mu        = zero<FLOATTYPE>();  /* damping constant */
	FLOATTYPE jacTe_inf = zero<FLOATTYPE>();  /* ||J^T e||_inf    */
	FLOATTYPE pDp_eL2   = zero<FLOATTYPE>();  /* ||e(p+Dp)||_2    */
	FLOATTYPE Dp_L2     = std::numeric_limits<FLOATTYPE>::max();
	FLOATTYPE p_L2;
	int nu   = 2;
	int stop = 0;
	int nfev = 0;
	int njev = 0;
	int nlss = 0;

	/* variables for constrained LM */
	func_state<FLOATTYPE> fstate;
	FLOATTYPE alpha = FLOATTYPE(1e-4);
	FLOATTYPE beta  = FLOATTYPE(0.9);
	FLOATTYPE gamma = FLOATTYPE(0.99995);
	FLOATTYPE rho   = FLOATTYPE(1e-8);
	FLOATTYPE t;
	FLOATTYPE jacTeDp;
	FLOATTYPE tmin  = FLOATTYPE(1e-12);
	FLOATTYPE tming = FLOATTYPE(1e-18);          /* minimum step length for LS and PG steps */
	constexpr FLOATTYPE tini = one<FLOATTYPE>(); /* initial step length for LS and PG steps */
	//int nLMsteps   = 0;
	//int nLSsteps   = 0;
	//int nPGsteps   = 0;
	int gprevtaken = 0;

	fstate.n     = n;
	fstate.hx    = hx;
	fstate.x     = x;
	fstate.lb    = lb;
	fstate.ub    = ub;
	fstate.adata = adata;
	fstate.nfev  = &nfev;

	/* see if starting point is within the feasible set */
	std::copy( p, p + m, pDp );
	boxProject( p, lb, ub, m ); /* project to feasible set */
	for( int i = 0; i < m; ++i )
	{
		if( pDp[i] != p[i] )
		{
			std::cerr << "Warning: component " << i << " of starting point not feasible in levmar_bc_der()! [" << pDp[i]
					  << " projected to " << p[i] << "]\n";
		}
	}

	/* compute e = x - f(p) and its L2 norm */
	(*func)( p, hx, m, n, adata );
	nfev=1;

	/* ### e = x-hx, p_eL2 = ||e(p)||_2 */
	FLOATTYPE       p_eL2      = levmar_L2nrmxmy( e, x, hx, n );
	const FLOATTYPE init_p_eL2 = p_eL2;

	if( !LMPP_FINITE(p_eL2) ) stop = 7; /* stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error */

	if( dscl )
	{	/* scale starting point and constraints */
		for( int i = m-1; i >= 0; --i ) p[i] /= dscl[i];
		boxScale( lb, ub, dscl, m, 1 );
	}

	int issolved;
	std::vector<char> solver_buffer;
	int iter_n = 0;
	for( ; iter_n < itmax && !stop; ++iter_n ) /* outter loop */
	{
		/* Note that p and e have been updated at a previous iteration */
		if( p_eL2 <= eps3 ) /* error is small */
		{
			stop = 6; /* stopped by small ||e||_2 */
			break; /* outter loop */
		}

		/* Compute the Jacobian J at p,  J^T J,  J^T e,  ||J^T e||_inf and ||p||^2.
		 * Since J^T J is symmetric, its computation can be sped up by computing
		 * only its upper triangular part and copying it to the lower part
		 */

		if( !dscl )
		{
			(*jacf)( p, jac, m, n, adata );
			++njev;
		}
		else
		{
			for( int i = m-1; i >= 0; --i )
			{
				sp_pDp[i] = p[i] * dscl[i];
			}
			(*jacf)( sp_pDp.data(), jac, m, n, adata );
			++njev;

			/* compute jac*D */
			for( int i = n-1; i >= 0; --i )
			{
				FLOATTYPE *jacim = jac + i*m;
				for( int j = m-1; j >= 0; --j )
				{
					jacim[j] *= dscl[j]; // jac[i*m+j] *= dscl[j];
				}
			}
		}

		/* J^T J, J^T e */
		if( nm < __BLOCKSZ__SQ ) // this is a small problem
		{                        /* J^T*J_ij = \sum_l J^T_il * J_lj = \sum_l J_li * J_lj.
		                          * Thus, the product J^T J can be computed using an outer loop for
		                          * l that adds J_li*J_lj to each element ij of the result. Note that
		                          * with this scheme, the accesses to J and JtJ are always along rows,
		                          * therefore induces less cache misses compared to the straightforward
		                          * algorithm for computing the product (i.e., l loop is innermost one).
		                          * A similar scheme applies to the computation of J^T e.
		                          * However, for large minimization problems (i.e., involving a large number
		                          * of unknowns and measurements) for which J/J^T J rows are too large to
		                          * fit in the L1 cache, even this scheme incures many cache misses. In
		                          * such cases, a cache-efficient blocking scheme is preferable.
		                          *
		                          * Thanks to John Nitao of Lawrence Livermore Lab for pointing out this
		                          * performance problem.
		                          *
		                          * Note that the non-blocking algorithm is faster on small
		                          * problems since in this case it avoids the overheads of blocking. 
		                          */
			std::fill( jacTjac, jacTjac + m*m, zero<FLOATTYPE>() );
			std::fill( jacTe,   jacTe   + m,   zero<FLOATTYPE>() );

			for( int l = n-1; l >= 0; --l )
			{
				FLOATTYPE* jaclm = jac + l*m;
				for( int i = m-1; i >= 0; --i )
				{
					FLOATTYPE* jacTjacim = jacTjac + i*m;
					FLOATTYPE  alpha     = jaclm[i]; // jac[l*m+i];
					for( int j = i; j >= 0; --j ) /* j <= i computes lower triangular part only */
					{
						jacTjacim[j] += jaclm[j] * alpha; //jacTjac[i*m+j] += jac[l*m+j] * alpha
					}
					/* J^T e */
					jacTe[i] += alpha * e[l];
				}
			}

			for( int i = m-1; i >= 0; --i ) /* copy to upper part */
			{
				for( int j = i+1; j < m; ++j )
				{
					jacTjac[i*m+j] = jacTjac[j*m+i];
				}
			}
		}
		else // this is a large problem
		{    /* Cache efficient computation of J^T J based on blocking
		      */
			levmar_trans_mat_mat_mult( jac, jacTjac, n, m );

			/* cache efficient computation of J^T e */
			std::fill( jacTe, jacTe + m, zero<FLOATTYPE>() );

			for( int i = 0; i < n; ++i )
			{
				FLOATTYPE *jacrow   = jac + i*m;
				const FLOATTYPE aux = e[i];
				for( int l = 0; l < m; ++l )
				{
					jacTe[l] += jacrow[l] * aux;
				}
			}
		}

		/* Compute ||J^T e||_inf and ||p||^2. Note that ||J^T e||_inf
		 * is computed for free (i.e. inactive) variables only. 
		 * At a local minimum, if p[i]==ub[i] then g[i]>0;
		 * if p[i]==lb[i] g[i]<0; otherwise g[i]=0 
		 */
		p_L2 = jacTe_inf = 0.0;
		int j         = 0;
		int numactive = 0;
		for( int i = 0; i < m; ++i )
		{
			if( ub && p[i] == ub[i] )
			{
				++numactive;
				if( jacTe[i] > zero<FLOATTYPE>() ) ++j;
			}
			else if( lb && p[i] == lb[i] )
			{
				++numactive;
				if( jacTe[i] < zero<FLOATTYPE>() ) ++j;
			}
			else
			{
				jacTe_inf = std::max( jacTe_inf, std::abs(jacTe[i]) );
			}

			diag_jacTjac[i] = jacTjac[i*m+i]; /* save diagonal entries so that augmentation can be later canceled */
			p_L2 += sq(p[i]);
		}
		//p_L2 = std::sqrt(p_L2);

	#if 0
		if( !(iter_n%100) )
		{
			std::cout << "Current estimate: ";
			for( int i = 0; i < m; ++i )
				std::cout << std::setw(10) << std::setprecision(9) << p[i];
			std::cout << std::setw(10) << std::setprecision(9) << "-- errors " << jacTe_inf << " " << p_eL2
					  << ", #active " << numactive << " [" << j << "]\n";
		}
	#endif

		/* check for convergence */
		if( j == numactive && jacTe_inf <= eps1 )
		{
			Dp_L2 = zero<FLOATTYPE>(); /* no increment for p in this case */
			stop = 1; /* stopped by small gradient J^T e */
			break;
		}

		/* compute initial damping factor */
		if( iter_n == 0 )
		{
			if( !lb && !ub ) /* no bounds */
			{
				mu = *std::max_element( diag_jacTjac, diag_jacTjac + m ) * tau;
			}
			else
			{
				mu = FLOATTYPE(0.5) * tau * p_eL2; /* use Kanzow's starting mu */
			}
		}

		/* determine increment using a combination of adaptive damping, line search and projected gradient search */
		while(1) /* inner loop */
		{
			/* augment normal equations */
			for( int i = 0; i < m; ++i )
			{
				jacTjac[i*m+i] += mu;
			}

			/* solve augmented equations */
		#ifdef LMPP_HAVE_LAPACK
			/* 6 alternatives are available: LU, Cholesky, LDLt, 2 variants of QR decomposition and SVD.
			 * From the serial solvers, Cholesky is the fastest but might occasionally be inapplicable due to numerical round-off;
			 * QR is slower but more robust; SVD is the slowest but most robust; LU is quite robust but
			 * slower than LDLt; LDLt offers a good tradeoff between robustness and speed
			 */

			//issolved = Ax_eq_b_Chol( jacTjac, jacTe, Dp, m, solver_buffer ); ++nlss;
			//issolved = Ax_eq_b_QR  ( jacTjac, jacTe, Dp, m, solver_buffer ); ++nlss;
			//issolved = Ax_eq_b_QRLS( jacTjac, jacTe, Dp, m, m, solver_buffer ); ++nlss;
			issolved = Ax_eq_b_BK  ( jacTjac, jacTe, Dp, m, solver_buffer ); ++nlss;
			//issolved = Ax_eq_b_LU  ( jacTjac, jacTe, Dp, m, solver_buffer ); ++nlss;
			//issolved = Ax_eq_b_SVD ( jacTjac, jacTe, Dp, m, solver_buffer ); ++nlss;

		#else /* No LAPACK --LMPP_HAVE_LAPACK */
			/* use the LU included with levmar */
			issolved = Ax_eq_b_LU( jacTjac, jacTe, Dp, m ); ++nlss;
		#endif /* LMPP_HAVE_LAPACK */

			if( issolved )
			{
				for( int i = 0; i < m; ++i )
				{
					pDp[i] = p[i] + Dp[i];
				}

				/* compute p's new estimate and ||Dp||^2 */
				boxProject( pDp, lb, ub, m ); /* project to feasible set */
				Dp_L2 = zero<FLOATTYPE>();
				for( int i = 0; i < m; ++i )
				{
					Dp[i]  = pDp[i] - p[i];
					Dp_L2 += sq(Dp[i]);
				}
				//Dp_L2 = std::sqrt(Dp_L2);

				if( Dp_L2 <= eps2_sq * p_L2 ) /* relative change in p is small, stop */
				{
					stop = 2; /* stopped by small Dp */
					break; /* inner loop */
				}

				if( Dp_L2 >= (p_L2 + eps2)/LMPP_SING_EPS ) /* almost singular */
				{
					stop = 4; /* singular matrix. Should restart from current p with increased mu */
					break; /* inner loop */
				}

				if( !dscl )
				{
					(*func)( pDp, hx, m, n, adata ); /* evaluate function at p + Dp */
					++nfev;
				}
				else
				{
					for( int i = m-1; i >= 0; --i )
					{
						sp_pDp[i] = pDp[i] * dscl[i];
					}

					(*func)( sp_pDp.data(), hx, m, n, adata); /* evaluate function at p + Dp */
					++nfev;
				}

				/* ### hx = x - hx, pDp_eL2 = ||hx|| */
			#if 1
				pDp_eL2 = levmar_L2nrmxmy( hx, x, hx, n );
			#else
				pDp_eL2 = zero<FLOATTYPE>();
				for( int i = 0; i < n; ++i ) /* compute ||e(pDp)||_2 */
				{
					hx[i]    =  x[i] - hx[i];
					pDp_eL2 += sq(hx[i]);
				}
			#endif

				/* the following test ensures that the computation of pDp_eL2 has not overflowed.
				 * Such an overflow does no harm here, thus it is not signalled as an error
				 */
				if( !LMPP_FINITE(pDp_eL2) && !LMPP_FINITE(lm_nrm2(n, hx)) )
				{
					stop = 7; /* stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error */
					break; /* inner loop */
				}

				if( pDp_eL2 <= gamma * p_eL2 )
				{
					FLOATTYPE dL = zero<FLOATTYPE>();
					for( int i = 0; i < m; ++i )
					{
						dL += Dp[i] * (mu * Dp[i] + jacTe[i]);
					}

				#if 1
					if( dL > 0.0 )
					{
						FLOATTYPE dF  = p_eL2 - pDp_eL2;
						FLOATTYPE aux = two<FLOATTYPE>() * dF / dL - one<FLOATTYPE>();
						aux           = one<FLOATTYPE>() - aux*aux*aux;
						mu           *= std::max( aux, FLOATTYPE(1./3.) );
					}
					else
					{
						FLOATTYPE aux = FLOATTYPE(0.1) * pDp_eL2; /* pDp_eL2 is the new p_eL2 */
						mu = std::min( mu, aux );
					}
				#else
					FLOATTYPE aux = FLOATTYPE(0.1) * pDp_eL2; /* pDp_eL2 is the new p_eL2 */
					mu = std::min( mu, aux );
				#endif

					nu=2;

					std::copy( pDp, pDp + m, p ); /* update p's estimate  */
					std::copy( hx,  hx  + n, e ); /* update e and ||e||_2 */


					p_eL2      = pDp_eL2;
					gprevtaken = 0;
					//++nLMsteps;
					break;
				}
				/* note that if the LM step is not taken, code falls through to the LM line search below */
			}
			else
			{
				/* the augmented linear system could not be solved, increase mu */

				mu *= nu;
				{
					int nu2 = nu<<1; // 2*nu;
					if( nu2 <= nu ) /* nu has wrapped around (overflown). Thanks to Frank Jordan for spotting this case */
					{
						stop = 5; /* no further error reduction is possible. Should restart with increased mu */
						break;
					}
					nu = nu2;
				}

				for( int i = 0; i < m; ++i ) /* restore diagonal J^T J entries */
				{
					jacTjac[i*m+i] = diag_jacTjac[i];
				}

				continue; /* solve again with increased nu */
			}

			/* if this point is reached, the LM step did not reduce the error;
			 * see if it is a descent direction
			 */

			/* negate jacTe (i.e. g) & compute g^T * Dp */
			jacTeDp = zero<FLOATTYPE>();
			for( int i = 0; i < m; ++i )
			{
				jacTe[i] = -jacTe[i];
				jacTeDp += jacTe[i] * Dp[i];
			}

			bool do_lnsrch = jacTeDp <= -rho * std::pow( Dp_L2, FLOATTYPE(2.1)/two<FLOATTYPE>() );
			if( do_lnsrch )
			{
				/* Dp is a descent direction; do a line search along it */
			#if 1
				/* use Schnabel's backtracking line search; it requires fewer "func" evaluations */
				{
					int mxtake, iretcd;
					FLOATTYPE steptl = FLOATTYPE(1e3) * std::sqrt( std::numeric_limits<FLOATTYPE>::epsilon() );
					FLOATTYPE stepmx = FLOATTYPE(1e3) * std::max( std::sqrt(p_L2), one<FLOATTYPE>() );

					lnsrch( m, p, p_eL2, jacTe, Dp, alpha, pDp, &pDp_eL2, func, &fstate,
							&mxtake, &iretcd, stepmx, steptl, dscl ); /* NOTE: LNSRCH() updates hx */
					if( iretcd != 0 || !LMPP_FINITE(pDp_eL2) )
					{
						do_lnsrch = false; /* handle LNSRCH() failures below... */
					}
				}
			#else
				/* use the simpler (but slower!) line search described by Kanzow et al */
				for( t = tini; t > tmin; t *= beta )
				{
					for( int i = 0; i < m; ++i )
					{
						pDp[i] = p[i] + t * Dp[i];
					}
					boxProject( pDp, lb, ub, m ); /* project to feasible set */

					if( !dscl )
					{
						(*func)( pDp, hx, m, n, adata ); /* evaluate function at p + t*Dp */
						++nfev;
					}
					else
					{
						for( i = m-1; i >= 0; --i )
						{
							sp_pDp[i] = pDp[i] * dscl[i];
						}
						(*func)( sp_pDp.data(), hx, m, n, adata ); /* evaluate function at p + t*Dp */
						++nfev;
					}

					/* compute ||e(pDp)||_2              */
					/* ### hx = x - hx, pDp_eL2 = ||hx|| */
				#if 1
					pDp_eL2 = levmar_L2nrmxmy( hx, x, hx, n );
				#else
					pDp_eL2 = zero<FLOATTYPE>();
					for( int i = 0; i < n; ++i )
					{
						hx[i]    = x[i] - hx[i];
						pDp_eL2 += sq(hx[i]);
					}
				#endif /* ||e(pDp)||_2 */
					if( !LMPP_FINITE(pDp_eL2) ) /* treat as line search failure */
					{
						do_lnsrch = false;
					}
					else
					{
						if( pDp_eL2 <= p_eL2 + two<FLOATTYPE>() * t * alpha * jacTeDp )
							break;
					}
				}
			#endif /* line search alternatives */

				if( do_lnsrch )
				{
					//++nLSsteps;
					gprevtaken=0;
				}

				/* NOTE: new estimate for p is in pDp, associated error in hx and its norm in pDp_eL2.
				 * These values are used below to update their corresponding variables 
				 */
			}

			if( !do_lnsrch )
			{
				/* Note that this point can also be reached when lnsrch() fails. */

				/* jacTe has been negated above. Being a descent direction, it is next used
				* to make a projected gradient step
				*/

				/* compute ||g|| */
				FLOATTYPE aux = FLOATTYPE(100.0) / ( one<FLOATTYPE>() + lm_nrm2( m, jacTe ) );
				/* guard against poor scaling & large steps; see (3.50) in C.T. Kelley's book */
				FLOATTYPE t0 = std::min( aux, tini );
				bool grad_search_ok = false;
				/* if the previous step was along the gradient descent, try to use the t employed in that step */
				for( t = (gprevtaken ? t : t0); t > tming; t *= beta )
				{
					for( int i = 0; i < m; ++i )
					{
						pDp[i] = p[i] - t * jacTe[i];
					}
					boxProject( pDp, lb, ub, m ); /* project to feasible set */
					Dp_L2 = zero<FLOATTYPE>();
					for( int i = 0; i < m; ++i )
					{
						Dp[i]  = pDp[i] - p[i];
						Dp_L2 += sq(Dp[i]);
					}

					if( !dscl )
					{
						(*func)( pDp, hx, m, n, adata ); /* evaluate function at p - t*g */
						++nfev;
					}
					else
					{
						for( int i = m-1; i >= 0; --i )
						{
							sp_pDp[i] = pDp[i] * dscl[i];
						}
						(*func)( sp_pDp.data(), hx, m, n, adata ); /* evaluate function at p - t*g */
						++nfev;
					}

					/* compute ||e(pDp)||_2 */
					/* ### hx = x - hx, pDp_eL2 = ||hx|| */
				#if 1
					pDp_eL2 = levmar_L2nrmxmy( hx, x, hx, n );
				#else
					pDp_eL2 = zero<FLOATTYPE>();
					for( int i = 0, ; i < n; ++i )
					{
						hx[i]    = x[i] - hx[i];
						pDp_eL2 += sq(hx[i]);
					}
				#endif
					/* the following test ensures that the computation of pDp_eL2 has not overflowed.
					 * Such an overflow does no harm here, thus it is not signalled as an error
					 */
					if( !LMPP_FINITE(pDp_eL2) && !LMPP_FINITE(lm_nrm2(n, hx)) )
					{
						stop = 7; /* stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error */
						goto breaknested;
					}

					/* compute ||g^T * Dp||. Note that if pDp has not been altered by projection
					 * (i.e. BOXPROJECT), jacTeDp = -t * ||g||^2
					 */
					jacTeDp = zero<FLOATTYPE>();
					for( int i = 0; i < m; ++i )
					{
						jacTeDp += jacTe[i] * Dp[i];
					}

					if( gprevtaken && pDp_eL2 <= p_eL2 + two<FLOATTYPE>() * FLOATTYPE(0.99999) * jacTeDp )
					{ /* starting t too small */
						t          = t0;
						gprevtaken = 0;
						continue;
					}

					if( pDp_eL2 <= p_eL2 + two<FLOATTYPE>() * alpha * jacTeDp )
					{
						grad_search_ok = true;
						break;
					}

					// sufficient decrease condition proposed by Kelley in (5.13)
					//if( pDp_eL2 <= p_eL2 - two<FLOATTYPE>() * alpha / t * Dp_L2 )
					//{
					//	grad_search_ok = true;
					//	break;
					//}
				}

				/* if this point is reached then the gradient line search has failed */
				if( grad_search_ok )
				{
					//++nPGsteps;
					gprevtaken=1;
					/* NOTE: new estimate for p is in pDp, associated error in hx and its norm in pDp_eL2 */
				}
				else
				{
					gprevtaken=0;
					break; /* inner loop */
				}
			} /* not lnsrch */

			/* update using computed values */
			Dp_L2 = zero<FLOATTYPE>();
			for( int i = 0; i < m; ++i )
			{
				Dp_L2 += sq(pDp[i] - p[i]);
			}
			//Dp_L2=sqrt(Dp_L2);

			if( Dp_L2 <= eps2_sq * p_L2 ) /* relative change in p is small, stop */
			{
				stop = 2; /* stopped by small Dp */
				break;
			}

			/* update p's estimate, e and ||e||_2 */
			std::copy( pDp, pDp + m, p );
			std::copy( hx,  hx  + n, e );
			p_eL2 = pDp_eL2;
			break; /* inner loop */
		} /* inner loop */
	} /* outter loop */

breaknested: /* NOTE: this point is also reached via an explicit goto! */

	if( iter_n >= itmax )
	{
		stop = 3; /* stopped by itmax */
	}

	for( int i = 0; i < m; ++i) /* restore diagonal J^T J entries */
		jacTjac[i*m+i]=diag_jacTjac[i];

	if( info )
	{
		info[0] = init_p_eL2;
		info[1] = p_eL2;
		info[2] = jacTe_inf;
		info[3] = Dp_L2;
		info[4] = mu / (*std::max_element( diag_jacTjac, diag_jacTjac + m ));
		info[5] = (FLOATTYPE)iter_n;
		info[6] = (FLOATTYPE)stop;
		info[7] = (FLOATTYPE)nfev;
		info[8] = (FLOATTYPE)njev;
		info[9] = (FLOATTYPE)nlss;
	}

	/* covariance matrix */
	if( covar )
	{
		for( int i = 0; i < m; ++i ) /* restore diagonal J^T J entries */
		{
			jacTjac[i*m+i] = diag_jacTjac[i];
		}
		levmar_covar<FLOATTYPE>( jacTjac, covar, p_eL2, m, n );

		if( dscl ) /* correct for the scaling */
		{
			for( int i = m-1; i >= 0; --i )
			{
				for( int j = m-1; j >= 0; --j )
				{
					covar[i*m+j] *= dscl[i] * dscl[j];
				}
			}
		}
	}
	/* covariance matrix */

#if 0
	std::cout << nLMsteps << " LM steps, " << nLSsteps << " line search, " << nPGsteps << " projected gradient\n";
#endif

	if( dscl )
	{
		/* scale final point and constraints */
		for( int i = 0; i < m; ++i )
		{
			p[i] *= dscl[i];
		}
		boxScale( lb, ub, dscl, m, 0 );
	}

	return ( stop ==4 || stop == 7 ) ?  LMPP_ERROR : iter_n;
}


/* following struct & lmbc_dif_XXX functions won't be necessary if a true secant
 * version of levmar_bc_dif() is implemented...
 */
template<typename FLOATTYPE>
struct lmbc_dif_data
{
	int                    ffdif; // nonzero if forward differencing is used
	FPTR<FLOATTYPE>        func;
	std::vector<FLOATTYPE> hx;
	std::vector<FLOATTYPE> hxx;
	void*                  adata;
	FLOATTYPE              delta;
};

template<typename FLOATTYPE>
static void lmbc_dif_func( FLOATTYPE* p, FLOATTYPE* hx, int m, int n, void *data )
{
	lmbc_dif_data<FLOATTYPE>* dta = reinterpret_cast<lmbc_dif_data<FLOATTYPE>*>(data);
	/* call user-supplied function passing it the user-supplied data */
	(*(dta->func))( p, hx, m, n, dta->adata );
}

template<typename FLOATTYPE>
static void lmbc_dif_jacf( FLOATTYPE* p, FLOATTYPE* jac, int m, int n, void *data )
{
	lmbc_dif_data<FLOATTYPE>* dta = reinterpret_cast<lmbc_dif_data<FLOATTYPE>*>(data);

	if( dta->ffdif )
	{
		/* evaluate user-supplied function at p */
		(*(dta->func))( p, dta->hx.data(), m, n, dta->adata );
		levmar_fdif_forw_jac_approx( dta->func, p, dta->hx.data(), dta->hxx.data(), dta->delta, jac, m, n, dta->adata );
	}
	else
	{
		levmar_fdif_cent_jac_approx( dta->func, p, dta->hx.data(), dta->hxx.data(), dta->delta, jac, m, n, dta->adata );
	}
}


/* No Jacobian version of the levmar_bc_der() function above: the Jacobian is approximated with 
 * the aid of finite differences (forward or central, see the comment for the opts argument)
 * Ideally, this function should be implemented with a secant approach. Currently, it just calls
 * levmar_bc_der()
 */
template<typename FLOATTYPE>
int levmar_bc_dif_impl( FPTR<FLOATTYPE> func, /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
						FLOATTYPE* p,         /* I/O: initial parameter estimates. On output has the estimated solution */
						FLOATTYPE* x,         /* I: measurement vector. NULL implies a zero vector */
						int        m,         /* I: parameter vector dimension (i.e. #unknowns) */
						int        n,         /* I: measurement vector dimension */
						FLOATTYPE* lb,        /* I: vector of lower bounds. If NULL, no lower bounds apply */
						FLOATTYPE* ub,        /* I: vector of upper bounds. If NULL, no upper bounds apply */
						FLOATTYPE* dscl,      /* I: diagonal scaling constants. NULL implies no scaling */
						int        itmax,     /* I: maximum number of iterations */
						FLOATTYPE  opts[5],   /* I: opts[0-4] = minim. options [\mu, \epsilon1, \epsilon2, \epsilon3, \delta]. Respectively the
						                       * scale factor for initial \mu, stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2 and
						                       * the step used in difference approximation to the Jacobian. Set to NULL for defaults to be used.
						                       * If \delta<0, the Jacobian is approximated with central differences which are more accurate
						                       * (but slower!) compared to the forward differences employed by default. 
						                       */
						FLOATTYPE  info[LMPP_INFO_SZ],
						                      /* O: information regarding the minimization. Set to NULL if don't care
						                       * info[0]= ||e||_2 at initial p.
						                       * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
						                       * info[5]= # iterations,
						                       * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
						                       *                                 2 - stopped by small Dp
						                       *                                 3 - stopped by itmax
						                       *                                 4 - singular matrix. Restart from current p with increased mu 
						                       *                                 5 - no further error reduction is possible. Restart with increased mu
						                       *                                 6 - stopped by small ||e||_2
						                       *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
						                       * info[7]= # function evaluations
						                       * info[8]= # Jacobian evaluations
						                       * info[9]= # linear systems solved, i.e. # attempts for reducing error
						                       */
						FLOATTYPE* work,      /* working memory at least LM_BC_DIF_WORKSZ() reals large, allocated if NULL */
						FLOATTYPE* covar,     /* O: Covariance matrix corresponding to LS solution; mxm. Set to NULL if not needed. */
						void*      adata )    /* pointer to possibly additional data, passed uninterpreted to func.
						                       * Set to NULL if not needed
						                       */
{
	lmbc_dif_data<FLOATTYPE> data;
	int ret;

	//std::cerr << "\nWarning: current implementation of levmar_bc_dif() does not use a secant approach!\n\n";

	data.ffdif = opts ? opts[4] >= zero<FLOATTYPE>() : 1;
	data.func  = func;
	data.hx .resize(n);
	data.hxx.resize(n);
	data.adata = adata;
	data.delta = opts ? std::abs(opts[4]) : (FLOATTYPE)LMPP_DIFF_DELTA;

	ret = levmar_bc_der( lmbc_dif_func, lmbc_dif_jacf,
						 p, x, m, n, lb, ub, dscl, itmax, opts, info, work, covar, (void *)&data );

	if( info ) /* correct the number of function calls */
	{
		if( data.ffdif )
			info[7] += info[8] * (m+1); /* each Jacobian evaluation costs m+1 function calls */
		else
			info[7] += info[8] * (2*m); /* each Jacobian evaluation costs 2*m function calls */
	}

	return ret;
}


#ifdef LMPP_DBL_PREC
int levmar_bc_der( FPTR<double> func,
				   FPTR<double> jacf,
				   double*      p,
				   double*      x,
				   int          m,
				   int          n,
				   double*      lb,
				   double*      ub,
				   double*      dscl,
				   int          itmax,
				   double       opts[4],
				   double       info[LMPP_INFO_SZ],
				   double*      work,
				   double*      covar,
				   void*        adata )
{
	return levmar_bc_der_impl<double>( func, jacf, p, x, m, n, lb, ub, dscl, itmax, opts, info, work, covar, adata );
}

int levmar_bc_dif( FPTR<double> func,
				   double*      p,
				   double*      x,
				   int          m,
				   int          n,
				   double*      lb,
				   double*      ub,
				   double*      dscl,
				   int          itmax,
				   double       opts[4],
				   double       info[LMPP_INFO_SZ],
				   double*      work,
				   double*      covar,
				   void*        adata )
{
	return levmar_bc_dif_impl<double>( func, p, x, m, n, lb, ub, dscl, itmax, opts, info, work, covar, adata );
}
#endif // LMPP_DBL_PREC

#ifdef LMPP_SNGL_PREC
int levmar_bc_der( FPTR<float> func,
				   FPTR<float> jacf,
				   float*      p,
				   float*      x,
				   int         m,
				   int         n,
				   float*      lb,
				   float*      ub,
				   float*      dscl,
				   int         itmax,
				   float       opts[4],
				   float       info[LMPP_INFO_SZ],
				   float*      work,
				   float*      covar,
				   void*       adata )
{
	return levmar_bc_der_impl( func, jacf, p, x, m, n, lb, ub, dscl, itmax, opts, info, work, covar, adata );
}

int levmar_bc_dif( FPTR<float> func,
				   float*      p,
				   float*      x,
				   int         m,
				   int         n,
				   float*      lb,
				   float*      ub,
				   float*      dscl,
				   int         itmax,
				   float       opts[4],
				   float       info[LMPP_INFO_SZ],
				   float*      work,
				   float*      covar,
				   void*       adata )
{
	return levmar_bc_dif_impl( func, p, x, m, n, lb, ub, dscl, itmax, opts, info, work, covar, adata );
}
#endif // LMPP_SNGL_PREC
