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
	//  Copyright (C) 2004  Manolis Lourakis (lourakis at ics forth gr)
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
 * Levenberg-Marquardt nonlinear minimization.
 ********************************************************************************/

#include <iostream>
#include <cstdlib>
#include <algorithm>

#include "compiler_lmpp.h"
#include "config_lmpp.h"
#include "macro_defs.h"
#include "levmar.h"
#include "misc.h"
#include "Axb.h"

#define LMPP_SING_EPS FLOATTYPE(1e-24)


/*
 * This function seeks the parameter vector p that best describes the measurements vector x.
 * More precisely, given a vector function  func : R^m --> R^n with n >= m,
 * it finds p s.t. func(p) ~= x, i.e. the squared second order (i.e. L2) norm of
 * e = x-func(p) is minimized.
 *
 * This function requires an analytic Jacobian. In case the latter is unavailable,
 * use levmar_dif() bellow
 *
 * Returns the number of iterations (>=0) if successful, LMPP_ERROR if failed
 *
 * For more details, see K. Madsen, H.B. Nielsen and O. Tingleff's lecture notes on 
 * non-linear least squares at http://www.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf
 */
template<typename FLOATTYPE>
int levmar_der_impl( FPTR<FLOATTYPE>  func,  /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
					 FPTR<FLOATTYPE>  jacf,  /* function to evaluate the Jacobian \part x / \part p */
					 FLOATTYPE*       p,     /* I/O: initial parameter estimates. On output has the estimated solution */
					 const FLOATTYPE* x,     /* I: measurement vector. NULL implies a zero vector */
					 const int        m,     /* I: parameter vector dimension (i.e. #unknowns) */
					 const int        n,     /* I: measurement vector dimension */
					 const int        itmax, /* I: maximum number of iterations */
					 const FLOATTYPE* opts,  /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor
					                          * for initial \mu, stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2.
					                          * Set to NULL for defaults to be used
					                          */
					 FLOATTYPE*       info,  /* O: information regarding the minimization. Set to NULL if don't care
					                          * info[0] = ||e||_2 at initial p.
					                          * info[1-4] =[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
					                          * info[5] = # iterations,
					                          * info[6] = reason for terminating: 1 - stopped by small gradient J^T e
					                          *                                   2 - stopped by small Dp
					                          *                                   3 - stopped by itmax
					                          *                                   4 - singular matrix. Restart from current p with increased mu
					                          *                                   5 - no further error reduction is possible. Restart with increased mu
					                          *                                   6 - stopped by small ||e||_2
					                          *                                   7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
					                          * info[7] = # function evaluations
					                          * info[8] = # Jacobian evaluations
					                          * info[9] = # linear systems solved, i.e. # attempts for reducing error
					                          */
					 FLOATTYPE*       work,  /* working memory at least LM_DER_WORKSZ() reals large, allocated if NULL */
					 FLOATTYPE*       covar, /* O: Covariance matrix corresponding to LS solution; mxm. Set to NULL if not needed. */
					 void*            adata ) /* pointer to possibly additional data, passed uninterpreted to func & jacf.
					                           * Set to NULL if not needed
					                           */
{
	if( n < m )
	{
		std::cerr << "levmar_der(): cannot solve a problem with fewer measurements [" << n << "] than unknowns [" << m << "]\n";
		return LMPP_ERROR;
	}

	if( !jacf )
	{
		std::cerr << "No function specified for computing the Jacobian in levmar_der()\n";
		std::cerr << "If no such function is available, use levmar_dif() rather than levmar_der()\n";
		return LMPP_ERROR;
	}

	const int nm = n*m;

	const FLOATTYPE tau     = opts ? opts[0] : FLOATTYPE(LMPP_INIT_MU);
	const FLOATTYPE eps1    = opts ? opts[1] : FLOATTYPE(LMPP_STOP_THRESH);
	const FLOATTYPE eps2    = opts ? opts[2] : FLOATTYPE(LMPP_STOP_THRESH);
	const FLOATTYPE eps3    = opts ? opts[3] : FLOATTYPE(LMPP_STOP_THRESH);
	const FLOATTYPE eps2_sq = eps2 * eps2;

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
	int njev = 0;
	int nlss = 0;

	/* compute e = x - f(p) and its L2 norm */
	(*func)( p, hx, m, n, adata );
	int nfev = 1;

	/* ### e = x-hx, p_eL2 = ||e(p)||_2 */
	FLOATTYPE       p_eL2      = levmar_L2nrmxmy( e, x, hx, n );
	const FLOATTYPE init_p_eL2 = p_eL2;

	if( !LMPP_FINITE(p_eL2) ) stop = 7; /* stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error */

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
		(*jacf)( p, jac, m, n, adata ); ++njev;

		/* J^T J, J^T e */
		if( nm < __BLOCKSZ__SQ ) // this is a small problem
		{					     /* J^T*J_ij = \sum_l J^T_il * J_lj = \sum_l J_li * J_lj.
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

			/* looping downwards saves a few computations */
			for( int l = n-1; l >= 0; --l )
			{
				FLOATTYPE* jaclm = jac + l*m;
				for( int i = m-1; i >= 0; --i )
				{
					FLOATTYPE* jacTjacim = jacTjac + i*m;
					FLOATTYPE  alpha     = jaclm[i]; // jac[l*m+i];
					for( int j = i; j >= 0; --j ) /* j <= i computes lower triangular part only */
					{
						jacTjacim[j] += jaclm[j] * alpha; // jacTjac[i*m+j] += jac[l*m+j] * alpha
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
		{    /* Cache efficient computation of J^T J based on blocking */
			levmar_trans_mat_mat_mult<FLOATTYPE>( jac, jacTjac, n, m );

			/* cache efficient computation of J^T e */
			std::fill( jacTe, jacTe + m, zero<FLOATTYPE>() );

			for( int i = 0; i < n; ++i )
			{
				FLOATTYPE  aux    = e[i];
				FLOATTYPE* jacrow = jac + i*m;
				for( int l = 0; l < m; ++l )
				{
					jacTe[l] += jacrow[l] * aux;
				}
			}
		}

		/* Compute ||J^T e||_inf and ||p||^2 */
		p_L2 = jacTe_inf = zero<FLOATTYPE>();
		for( int i = 0; i < m; ++i )
		{
			jacTe_inf       = std::max( jacTe_inf, std::abs(jacTe[i]) );
			diag_jacTjac[i] = jacTjac[i*m+i]; /* save diagonal entries so that augmentation can be later canceled */
			p_L2           += p[i] * p[i];
		}
		// p_L2 = sqrt(p_L2);

	#if 0
		if( !(iter_n%100) )
		{
			std::cout << "Current estimate: ";
			for(int i = 0; i < m; ++i )
			{
				std::cout << p[i] << " ";
			}
			std::cout << "-- errors " << jacTe_inf << " " << p_eL2 << "\n";
		}
	#endif

		/* check for convergence */
		if( jacTe_inf <= eps1 )
		{
			Dp_L2 = zero<FLOATTYPE>(); /* no increment for p in this case */
			stop = 1; /* stopped by small gradient J^T e */
			break;
		}

		/* compute initial damping factor */
		if( iter_n == 0 )
		{
			mu = *std::max_element( diag_jacTjac, diag_jacTjac + m ) * tau;
		}

		/* determine increment using adaptive damping */
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

		#else
			/* use the LU included with levmar */
			issolved = Ax_eq_b_LU( jacTjac, jacTe, Dp, m ); ++nlss;
		#endif /* LMPP_HAVE_LAPACK */

			if( issolved )
			{
				/* compute p's new estimate and ||Dp||^2 */
				Dp_L2 = zero<FLOATTYPE>();
				for( int i = 0; i < m; ++i )
				{
					pDp[i] = p[i] + Dp[i];
					Dp_L2 += sq(Dp[i]);
				}
				//Dp_L2=sqrt(Dp_L2);

				if( Dp_L2 <= eps2_sq * p_L2 ) /* relative change in p is small, stop */
				//if( Dp_L2 <= eps2 * (p_L2 + eps2) ) /* relative change in p is small, stop */
				{
					stop = 2; /* stopped by small Dp */
					break; /* inner loop */
				}

				if( Dp_L2 >= (p_L2 + eps2)/LMPP_SING_EPS ) /* almost singular */
				{
					stop = 4; /* singular matrix. Should restart from current p with increased mu */
					break; /* inner loop */
				}

				(*func)( pDp, hx, m, n, adata ); ++nfev; /* evaluate function at p + Dp */

				/* compute ||e(pDp)||_2 */
				/* ### hx=x-hx, pDp_eL2=||hx|| */
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

				if( !LMPP_FINITE(pDp_eL2) ) /* sum of squares is not finite, most probably due to a user error.
										     * This check makes sure that the inner loop does not run indefinitely.
										     * Thanks to Steve Danauskas for reporting such cases
										     */
				{
					stop = 7; /* stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error */
					break; /* inner loop */
				}

				FLOATTYPE dL = zero<FLOATTYPE>();
				for( int i = 0; i < m; ++i )
				{
					dL += Dp[i] * (mu * Dp[i] + jacTe[i]);
				}

				FLOATTYPE dF = p_eL2 - pDp_eL2;

				if( dL > zero<FLOATTYPE>() && dF > zero<FLOATTYPE>() ) /* reduction in error, increment is accepted */
				{
					FLOATTYPE aux = two<FLOATTYPE>() * dF / dL - one<FLOATTYPE>();
					aux           = one<FLOATTYPE>() - aux*aux*aux;
					mu           *= std::max( aux, FLOATTYPE(1./3.) );
					nu            = 2;

					std::copy( pDp, pDp + m, p ); /* update p's estimate  */
					std::copy( hx,  hx  + n, e ); /* update e and ||e||_2 */

					p_eL2 = pDp_eL2;
					break; /* inner loop */
				}
			}

			/* if this point is reached, either the linear system could not be solved or
			 * the error did not reduce; in any case, the increment must be rejected
			 */

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
		} /* inner loop */
	} /* outter loop */

	if( iter_n >= itmax )
	{
		stop = 3; /* stopped by itmax */
	}

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
	}

	return ( stop ==4 || stop == 7 ) ?  LMPP_ERROR : iter_n;
}


/* Secant version of the LEVMAR_DER() function above: the Jacobian is approximated with 
 * the aid of finite differences (forward or central, see the comment for the opts argument)
 */
template<typename FLOATTYPE>
int levmar_dif_impl( FPTR<FLOATTYPE>  func,  /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
					 FLOATTYPE*       p,     /* I/O: initial parameter estimates. On output has the estimated solution */
					 const FLOATTYPE* x,     /* I: measurement vector. NULL implies a zero vector */
					 const int        m,     /* I: parameter vector dimension (i.e. #unknowns) */
					 const int        n,     /* I: measurement vector dimension */
					 const int        itmax, /* I: maximum number of iterations */
					 const FLOATTYPE* opts,  /* I: opts[0-4] = minim. options [\mu, \epsilon1, \epsilon2, \epsilon3, \delta]. Respectively the
					                          *    scale factor for initial \mu, stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2 and
					                          *    the step used in difference approximation to the Jacobian. Set to NULL for defaults to be used.
					                          *    If \delta<0, the Jacobian is approximated with central differences which are more accurate
					                          *    (but slower!) compared to the forward differences employed by default. 
					                          */
					 FLOATTYPE*       info,  /* O: information regarding the minimization. Set to NULL if don't care
					                          * info[0] = ||e||_2 at initial p.
					                          * info[1-4] =[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
					                          * info[5] = # iterations,
					                          * info[6] = reason for terminating: 1 - stopped by small gradient J^T e
					                          *                                   2 - stopped by small Dp
					                          *                                   3 - stopped by itmax
					                          *                                   4 - singular matrix. Restart from current p with increased mu
					                          *                                   5 - no further error reduction is possible. Restart with increased mu
					                          *                                   6 - stopped by small ||e||_2
					                          *                                   7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
					                          * info[7] = # function evaluations
					                          * info[8] = # Jacobian evaluations
					                          * info[9] = # linear systems solved, i.e. # attempts for reducing error
					                          */
					 FLOATTYPE*       work,  /* working memory at least LM_DIF_WORKSZ() reals large, allocated if NULL */
					 FLOATTYPE*       covar, /* O: Covariance matrix corresponding to LS solution; mxm. Set to NULL if not needed. */
					 void*            adata ) /* pointer to possibly additional data, passed uninterpreted to func & jacf.
					                           * Set to NULL if not needed
					                           */
{
	if( n < m )
	{
		std::cerr << "levmar_der(): cannot solve a problem with fewer measurements [" << n << "] than unknowns [" << m << "]\n";
		return LMPP_ERROR;
	}

	const int nm = n*m;

	const FLOATTYPE tau     = opts ? opts[0] : FLOATTYPE(LMPP_INIT_MU);
	const FLOATTYPE eps1    = opts ? opts[1] : FLOATTYPE(LMPP_STOP_THRESH);
	const FLOATTYPE eps2    = opts ? opts[2] : FLOATTYPE(LMPP_STOP_THRESH);
	const FLOATTYPE eps3    = opts ? opts[3] : FLOATTYPE(LMPP_STOP_THRESH);
	const FLOATTYPE delta   = std::abs(opts ? opts[4] : FLOATTYPE(LMPP_DIFF_DELTA));                     /* make positive             */
	const int using_ffdif   = int( (opts ? opts[4] : FLOATTYPE(LMPP_DIFF_DELTA)) >= zero<FLOATTYPE>() ); /* use forward differencing? */
	const FLOATTYPE eps2_sq = eps2 * eps2;

	std::unique_ptr<FLOATTYPE[]> buf;
	if( !work )
	{
		buf = std::make_unique<FLOATTYPE[]>( LMPP_DIF_WORKSZ(m, n) ); // worksz = 4*n+4*m + n*m + m*m;
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
	FLOATTYPE* wrk          = pDp          + m;   /*                    (n,1) */
	FLOATTYPE* wrk2         = wrk          + n;   /* (n,1), used only for holding a temporary e vector and when differentiating with central differences */

	FLOATTYPE mu        = zero<FLOATTYPE>(); /* damping constant */
	FLOATTYPE jacTe_inf = zero<FLOATTYPE>(); /* ||J^T e||_inf    */
	FLOATTYPE pDp_eL2   = zero<FLOATTYPE>(); /* ||e(p+Dp)||_2    */
	FLOATTYPE Dp_L2     = std::numeric_limits<FLOATTYPE>::max();
	FLOATTYPE p_L2;
	int nu     = 20; /* force computation of J */
	int stop   = 0;
	int njap   = 0;
	int updp   = 1;
	int updjac = 0;
	int newjac = 0;
	int nlss   = 0;
	int K      = std::max( m, 10 );

	/* compute e = x - f(p) and its L2 norm */
	(*func)( p, hx, m, n, adata );
	int nfev = 1;

	/* ### e = x - hx, p_eL2 = ||e(p)||_2 */
	FLOATTYPE       p_eL2      = levmar_L2nrmxmy( e, x, hx, n );
	const FLOATTYPE init_p_eL2 = p_eL2;

	if( !LMPP_FINITE(p_eL2) ) stop = 7; /* stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error */

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
		 * The symmetry of J^T J is again exploited for speed
		 */

		if( (updp && nu > 16) || updjac == K ) /* compute difference approximation to J */
		{
			if( using_ffdif ) /* use forward differences */
			{
				levmar_fdif_forw_jac_approx<FLOATTYPE>( func, p, hx, wrk, delta, jac, m, n, adata );
				++njap;
				nfev += m;
			}
			else /* use central differences */
			{
				levmar_fdif_cent_jac_approx<FLOATTYPE>( func, p, wrk, wrk2, delta, jac, m, n, adata );
				++njap;
				nfev += 2*m;
			}
			nu     = 2;
			updjac = 0;
			updp   = 0;
			newjac = 1;
		}

		if( newjac ) /* Jacobian has changed, recompute J^T J, J^t e, etc */
		{
			newjac = 0;
			/* J^T J, J^T e */
			if( nm <= __BLOCKSZ__SQ ) // this is a small problem
			{					      /* J^T*J_ij = \sum_l J^T_il * J_lj = \sum_l J_li * J_lj.
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

				/* looping downwards saves a few computations */
				for( int l = n-1; l >= 0; --l )
				{
					const FLOATTYPE* jaclm = jac + l*m;
					for( int i = m-1; i >= 0; --i )
					{
						FLOATTYPE*      jacTjacim = jacTjac + i*m;
						const FLOATTYPE alpha     = jaclm[i]; // jac[l*m+i];
						for( int j = i; j >= 0; --j ) /* j <= i computes lower triangular part only */
						{
							jacTjacim[j] += jaclm[j] * alpha; //jacTjac[i*m+j]+=jac[l*m+j]*alpha
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
			{	/* Cache efficient computation of J^T J based on blocking
				 */
				levmar_trans_mat_mat_mult<FLOATTYPE>( jac, jacTjac, n, m );

				/* cache efficient computation of J^T e */
				std::fill( jacTe, jacTe + m, zero<FLOATTYPE>() );

				for( int i = 0; i<n; ++i )
				{
					const FLOATTYPE* jacrow = jac + i*m;
					const FLOATTYPE  aux    = e[i];
					for( int l = 0; l < m; ++l )
					{
						jacTe[l] += jacrow[l] * aux;
					}
				}
			}

			/* Compute ||J^T e||_inf and ||p||^2 */
			p_L2 = jacTe_inf = zero<FLOATTYPE>();
			for( int i = 0; i < m; ++i )
			{
				jacTe_inf       = std::max( jacTe_inf, std::abs(jacTe[i]) );
				diag_jacTjac[i] = jacTjac[i*m+i]; /* save diagonal entries so that augmentation can be later canceled */
				p_L2           += p[i] * p[i];
			}
			//p_L2=sqrt(p_L2);
		} /* newjac */

	#if 0
		if( !(iter_n%100) )
		{
			std::cout << "Current estimate: ";
			for(int i = 0; i < m; ++i )
			{
				std::cout << p[i] << " ";
			}
			std::cout << "-- errors " << jacTe_inf << " " << p_eL2 << "\n";
		}
	#endif

		/* check for convergence */
		if( jacTe_inf <= eps1 )
		{
			Dp_L2 = zero<FLOATTYPE>(); /* no increment for p in this case */
			stop = 1; /* stopped by small gradient J^T e */
			break;
		}

		/* compute initial damping factor */
		if( iter_n == 0 )
		{
			mu = *std::max_element( diag_jacTjac, diag_jacTjac + m ) * tau;
		}

		/* determine increment using adaptive damping */

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
	#else
		/* use the LU included with levmar */
		issolved = Ax_eq_b_LU( jacTjac, jacTe, Dp, m ); ++nlss;
	#endif /* LMPP_HAVE_LAPACK */

		if( issolved )
		{
			/* compute p's new estimate and ||Dp||^2 */
			Dp_L2 = zero<FLOATTYPE>();
			for( int i = 0; i < m; ++i )
			{
				pDp[i] = p[i] + Dp[i];
				Dp_L2 += Dp[i] * Dp[i];
			}
			//Dp_L2=sqrt(Dp_L2);

			if( Dp_L2 <= eps2_sq * p_L2 ) /* relative change in p is small, stop */
			//if( Dp_L2 <= eps2 * (p_L2 + eps2) ) /* relative change in p is small, stop */
			{
				stop = 2; /* stopped by small Dp */
				break; /* inner loop */
			}

			if( Dp_L2 >= (p_L2 + eps2)/LMPP_SING_EPS ) /* almost singular */
			{
				stop = 4; /* singular matrix. Should restart from current p with increased mu */
				break; /* inner loop */
			}

			(*func)( pDp, wrk, m, n, adata ); ++nfev; /* evaluate function at p + Dp */

			/* compute ||e(pDp)||_2 */
			/* ### wrk2 = x - wrk, pDp_eL2 = ||wrk2|| */
		#if 1
			pDp_eL2 = levmar_L2nrmxmy( wrk2, x, wrk, n );
		#else
			pDp_eL2 = zero<FLOATTYPE>();
			for( int i = 0; i < n; ++i )
			{
				wrk2[i]  =    x[i] -  wrk[i];
				pDp_eL2 += wrk2[i] * wrk2[i];
		}
		#endif
			if( !LMPP_FINITE(pDp_eL2) ) /* sum of squares is not finite, most probably due to a user error.
										 * This check makes sure that the loop terminates early in the case
										 * of invalid input. Thanks to Steve Danauskas for suggesting it
										 */
			{
				stop = 7; /* stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error */
				break;
			}

			FLOATTYPE dF = p_eL2 - pDp_eL2;
			if( updp || dF > 0 ) /* update jac */
			{
				for( int i = 0; i < n; ++i )
				{
					FLOATTYPE aux = zero<FLOATTYPE>();
					for( int l = 0; l < m; ++l )
					{
						aux += jac[i*m+l] * Dp[l]; /* (J * Dp)[i] */
					}
					aux = (wrk[i] - hx[i] - aux) / Dp_L2; /* (f(p+dp)[i] - f(p)[i] - (J * Dp)[i]) / (dp^T*dp) */
					for( int j = 0; j < m; ++j )
					{
						jac[i*m+j] += aux * Dp[j];
					}
				}
				++updjac;
				newjac = 1;
			}

			FLOATTYPE dL = zero<FLOATTYPE>();
			for( int i = 0; i < m; ++i )
			{
				dL += Dp[i] * (mu*Dp[i] + jacTe[i]);
			}

			if( dL > zero<FLOATTYPE>() && dF > zero<FLOATTYPE>() ) /* reduction in error, increment is accepted */
			{
				FLOATTYPE aux = ( two<FLOATTYPE>() * dF / dL - one<FLOATTYPE>() );
				aux           = one<FLOATTYPE>() - aux*aux*aux;
				mu           *= std::max( aux, FLOATTYPE(1./3.) );
				nu            = 2;

				std::copy( pDp, pDp + m, p ); /* update p's estimate  */

				for( int i = 0; i < n; ++i ) /* update e, hx and ||e||_2 */
				{
					e[i]  = wrk2[i]; // x[i] - wrk[i];
					hx[i] = wrk[i];
				}
				p_eL2 = pDp_eL2;
				updp  = 1;
				continue;
			}
		} /* if issolved */

		/* if this point is reached, either the linear system could not be solved or
		 * the error did not reduce; in any case, the increment must be rejected
		 */

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
	} /* outter loop */

	if( iter_n >= itmax )
	{
		stop = 3; /* stopped by itmax */
	}

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
		info[8] = (FLOATTYPE)njap;
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
	}

	return ( stop ==4 || stop == 7 ) ?  LMPP_ERROR : iter_n;
}

#ifdef LMPP_DBL_PREC
int levmar_der( FPTR<double> func,
				FPTR<double> jacf,
				double*      p,
				double*      x,
				int          m,
				int          n,
				int          itmax,
				double*      opts,
				double*      info,
				double*      work,
				double*      covar,
				void*        adata )
{
	return levmar_der_impl<double>( func, jacf, p, x, m, n, itmax, opts, info, work, covar, adata );
}

int levmar_dif( FPTR<double> func,
				double*      p,
				double*      x,
				int          m,
				int          n,
				int          itmax,
				double*      opts,
				double*      info,
				double*      work,
				double*      covar,
				void*        adata )
{
	return levmar_dif_impl<double>( func, p, x, m, n, itmax, opts, info, work, covar, adata );
}

void levmar_chkjac( FPTR<double> func, FPTR<double> jacf, double* p, int m, int n, void* adata, double* err )
{
	levmar_chkjac_impl( func, jacf, p, m, n, adata, err );
}
#endif // LMPP_DBL_PREC

#ifdef LMPP_SNGL_PREC
int levmar_der( FPTR<float> func,
				FPTR<float> jacf,
				float*      p,
				float*      x,
				int         m,
				int         n,
				int         itmax,
				float*      opts,
				float*      info,
				float*      work,
				float*      covar,
				void*       adata )
{
	return levmar_der_impl<float>( func, jacf, p, x, m, n, itmax, opts, info, work, covar, adata );
}

int levmar_dif( FPTR<float> func,
				float*      p,
				float*      x,
				int         m,
				int         n,
				int         itmax,
				float*      opts,
				float*      info,
				float*      work,
				float*      covar,
				void*       adata )
{
	return levmar_dif_impl<float>( func, p, x, m, n, itmax, opts, info, work, covar, adata );
}

void levmar_chkjac( FPTR<float> func, FPTR<float> jacf, float* p, int m, int n, void* adata, float* err )
{
	levmar_chkjac_impl( func, jacf, p, m, n, adata, err );
}
#endif // LMPP_SNGL_PREC
