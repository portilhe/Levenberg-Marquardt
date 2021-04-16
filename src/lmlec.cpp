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

/*******************************************************************************
 * Wrappers for linearly constrained Levenberg-Marquardt minimization. The same
 * core code is used with appropriate #defines to derive single and double
 * precision versions, see also lmlec_core.c
 *******************************************************************************/

//#include <cstdio>
//#include <cstdlib>
//#include <cmath>
#include <vector>
#include <limits>
#include <iostream>
#include <algorithm>

#include "levmar.h"
#include "misc.h"
#include "external_wrappers.h"


#ifndef LMPP_HAVE_LAPACK

#ifdef _MSC_VER
#pragma message("Linearly constrained optimization requires LAPACK and was not compiled!")
#else
#warning Linearly constrained optimization requires LAPACK and was not compiled!
#endif // _MSC_VER

#else // LAPACK present

#if !defined(LMPP_DBL_PREC) && !defined(LMPP_SNGL_PREC)
#error At least one of LM_DBL_PREC, LM_SNGL_PREC should be defined!
#endif // LMPP_HAVE_LAPACK

#define LEVMAR_LEC_DER LM_ADD_PREFIX(levmar_lec_der)
#define LEVMAR_LEC_DIF LM_ADD_PREFIX(levmar_lec_dif)
#define LEVMAR_DER LM_ADD_PREFIX(levmar_der)
#define LEVMAR_DIF LM_ADD_PREFIX(levmar_dif)
#define LEVMAR_TRANS_MAT_MAT_MULT LM_ADD_PREFIX(levmar_trans_mat_mat_mult)
#define LEVMAR_COVAR LM_ADD_PREFIX(levmar_covar)
#define LEVMAR_FDIF_FORW_JAC_APPROX LM_ADD_PREFIX(levmar_fdif_forw_jac_approx)

#define GEQP3 LM_MK_LAPACK_NAME(geqp3)
#define ORGQR LM_MK_LAPACK_NAME(orgqr)
#define TRTRI LM_MK_LAPACK_NAME(trtri)

template<typename FLOATTYPE>
struct lmlec_data{
	FLOATTYPE*      c;
	FLOATTYPE*      Z;
	FLOATTYPE*      p;
	FLOATTYPE*      jac;
	int             ncnstr;
	FPTR<FLOATTYPE> func;
	FPTR<FLOATTYPE> jacf;
	void*           adata;
};

/*
 * This function implements an elimination strategy for linearly constrained
 * optimization problems. The strategy relies on QR decomposition to transform
 * an optimization problem constrained by Ax=b to an equivalent, unconstrained
 * one. Also referred to as "null space" or "reduced Hessian" method.
 * See pp. 430-433 (chap. 15) of "Numerical Optimization" by Nocedal-Wright
 * for details.
 *
 * A is mxn with m <= n and rank(A) = m
 * Two matrices Y and Z of dimensions nxm and nx(n-m) are computed from A^T so that
 * their columns are orthonormal and every x can be written as x = Y*b + Z*x_z =
 * c + Z*x_z, where c = Y*b is a fixed vector of dimension n and x_z is an
 * arbitrary vector of dimension n-m. Then, the problem of minimizing f(x)
 * subject to Ax = b is equivalent to minimizing f(c + Z*x_z) with no constraints.
 * The computed Y and Z are such that any solution of Ax = b can be written as
 * x = Y*x_y + Z*x_z for some x_y, x_z. Furthermore, A*Y is nonsingular, A*Z = 0
 * and Z spans the null space of A.
 *
 * The function accepts A, b and computes c, Y, Z. If b or c is NULL, c is not
 * computed. Also, Y can be NULL in which case it is not referenced.
 * The function returns LM_ERROR in case of error, A's computed rank if successful
 *
 */
template<typename FLOATTYPE>
static int lmlec_elim( FLOATTYPE* A, FLOATTYPE* b, FLOATTYPE* c, FLOATTYPE* Y, FLOATTYPE* Z, int m, int n )
{
	constexpr FLOATTYPE eps = std::numeric_limits<FLOATTYPE>::epsilon();

	if( m > n )
	{
		std::cerr << "matrix of constraints cannot have more rows than columns in lmlec_elim()!\n";
		return LMPP_ERROR;
	}

	const int tm     = n;
	const int tn     = m; // transpose dimensions
	const int mintmn = m;

	/* calculate required memory size */
	int info;
	int work_sz = -1; // workspace query. Optimal work size is returned in aux
	{
		FLOATTYPE aux;
		//lm_orgqr<FLOATTYPE>( &tm, &tm, &mintmn, nullptr, &tm, nullptr, &aux, &work_sz, &info);
		lm_geqp3<FLOATTYPE>( &tm, &tn, nullptr, &tm, nullptr, nullptr, &aux, &work_sz, &info);
		work_sz = (int)aux;
	}

	const int a_sz    = sq(tm); // tm*tn is enough for xgeqp3()
	const int jpvt_sz = tn;
	const int tau_sz  = mintmn;
	const int r_sz    = sq(mintmn); // actually smaller if a is not of full row rank
	const int Y_sz    = Y ? 0 : tm*tn;
	const int tot_sz  = (a_sz + tau_sz + r_sz + work_sz + Y_sz) * sizeof(FLOATTYPE) + jpvt_sz * sizeof(int); /* should be arranged in that order for proper doubles alignment */

	std::vector<char> buf; /* allocate a "big" memory chunk at once */
	buf.resize(tot_sz);

	FLOATTYPE* a    = reinterpret_cast<FLOATTYPE*>(buf.data());
	FLOATTYPE* tau  = a   + a_sz;
	FLOATTYPE* r    = tau + tau_sz;
	FLOATTYPE* work = r   + r_sz;

	int *jpvt;
	if( !Y )
	{
		Y    = work + work_sz;
		jpvt = reinterpret_cast<int*>(Y + Y_sz);
	}
	else
	{
		jpvt = reinterpret_cast<int*>( work + work_sz );
	}

	/* copy input array so that LAPACK won't destroy it. Note that copying is
	 * done in row-major order, which equals A^T in column-major
	 */
	std::copy( A, A + tm*tn, a );

	/* clear jpvt */
	std::fill( jpvt, jpvt + jpvt_sz, 0 );

	/* rank revealing QR decomposition of A^T*/
	lm_geqp3( &tm, &tn, a, &tm, jpvt, tau, work, &work_sz, &info );
	//dgeqpf_((int *)&tm, (int *)&tn, a, (int *)&tm, jpvt, tau, work, &info);
	/* error checking */
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_geqp3 in lmlec_elim()\n";
			exit(1);
		}
		else
		{
			std::cerr << "unknown LAPACK error for lm_geqp3 in lmlec_elim() [info=" << info << "]\n";
			return 0;
		}
		return LMPP_ERROR;
	}

	/* the upper triangular part of a now contains the upper triangle of the unpermuted R */
	FLOATTYPE aux = tm * FLOATTYPE(10.0) * eps * std::abs(a[0]); // threshold. tm is max(tm, tn)
	aux = std::max( aux, FLOATTYPE(1E-12) );                     // ensure that threshold is not too small

	/* compute A^T's numerical rank by counting the non-zeros in R's diagonal */
	int rank = 0;
	for( int i = 0; i < mintmn; ++i )
	{
		if( a[i*(tm+1)] > aux || a[i*(tm+1)] < -aux ) /* loop across R's diagonal elements */
		{
			++rank;
		}
		else
		{
			break; /* diagonal is arranged in absolute decreasing order */
		}
	}

	if( rank < tn )
	{
		std::cerr << "\nConstraints matrix in lmlec_elim() is not of full row rank (i.e. " << rank << " < " << tn << "d)!\n";
		std::cerr << "Make sure that you do not specify redundant or inconsistent constraints.\n\n";
		return LMPP_ERROR;
	}

	/* compute the permuted inverse transpose of R */
	/* first, copy R from the upper triangular part of a to the lower part of r (thus transposing it). R is rank x rank */
	for( int j = 0; j < rank; ++j )
	{
		for( int i = 0; i <= j; ++i )
		{
			r[j+i*rank] = a[i+j*tm];
		}
		for( int i = j+1; i < rank; ++i )
		{
			r[j+i*rank] = zero<FLOATTYPE>(); // upper part is zero
		}
	}
	/* r now contains R^T */

	/* compute the inverse */
	lm_trtri( "L", "N", &rank, r, &rank, &info );
	/* error checking */
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_trtri in lmlec_elim()\n";
			exit(1);
		}
		else
		{
			std::cerr << "A(" << info << "," << info << "is exactly zero for lm_trtri (singular matrix) in lmlec_elim()\n";
			return 0;
		}
		return LMPP_ERROR;
	}

	/* finally, permute R^-T using Y as intermediate storage */
	for( int j = 0; j < rank; ++j )
	{
		int k = jpvt[j]-1;
		for( int i = 0; i < rank; ++i )
		{
			Y[i+k*rank] = r[i+j*rank];
		}
	}

	std::copy( Y, Y + sq(rank), r ); // copy back to r
	/* resize a to be tm x tm, filling with zeroes */
	std::fill( a + tm*tn, a + sq(tm), zero<FLOATTYPE>() );

	/* compute Q in a as the product of elementary reflectors. Q is tm x tm */
	lm_orgqr( &tm, &tm, &mintmn, a, &tm, tau, work, &work_sz, &info );
	/* error checking */
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_orgqr in lmlec_elim()\n";
			exit(1);
		}
		else
		{
			std::cerr << "unknown LAPACK error for lm_orgqr in lmlec_elim() [info=" << info << "]\n";
			return 0;
		}
		return LMPP_ERROR;
	}

	/* compute Y=Q_1*R^-T*P^T. Y is tm x rank */
	for( int i = 0; i < tm; ++i )
	{
		for( int j = 0; j < rank; ++j )
		{
			FLOATTYPE aux = zero<FLOATTYPE>();
			for( int k = 0; k < rank; ++k )
			{
				aux += a[i+k*tm] * r[k+j*rank];
			}
			Y[i*rank+j] = aux;
		}
	}

	if( b && c )
	{
		/* compute c = Y*b */
		for( int i = 0; i < tm; ++i )
		{
			FLOATTYPE aux = zero<FLOATTYPE>();
			for( int j = 0; j < rank; ++j )
			{
				aux += Y[i*rank+j] * b[j];
			}
			c[i] = aux;
		}
	}

	/* copy Q_2 into Z. Z is tm x (tm-rank) */
	const int tmmr = tm - rank;
	for( int j = 0; j < tmmr; ++j )
	{
		const int k = j + rank;
		for( int i = 0; i < tm; ++i )
		{
			Z[i*tmmr+j] = a[i+k*tm];
		}
	}

	return rank;
}

/* constrained measurements: given pp, compute the measurements at c + Z*pp */
template<typename FLOATTYPE>
static void lmlec_func( FLOATTYPE* pp, FLOATTYPE* hx, int mm, int n, void *adata )
{
	lmlec_data<FLOATTYPE>* data = reinterpret_cast<lmlec_data<FLOATTYPE>*>(adata);

	int        m = mm + data->ncnstr;
	FLOATTYPE* c = data->c;
	FLOATTYPE* Z = data->Z;
	FLOATTYPE* p = data->p;
	/* p=c + Z*pp */
	for( int i = 0; i < m; ++i )
	{
		FLOATTYPE* Zimm = Z + i*mm;
		FLOATTYPE  sum_ = c[i];
		for( int j = 0; j < mm; ++j )
		{
			sum_ += Zimm[j] * pp[j]; // sum_ += Z[i*mm+j] * pp[j];
		}
		p[i] = sum_;
	}

	(*(data->func))( p, hx, m, n, data->adata );
}

/* constrained Jacobian: given pp, compute the Jacobian at c + Z*pp
* Using the chain rule, the Jacobian with respect to pp equals the
* product of the Jacobian with respect to p (at c + Z*pp) times Z
*/
template<typename FLOATTYPE>
static void lmlec_jacf( FLOATTYPE* pp, FLOATTYPE* jacjac, int mm, int n, void *adata )
{
	lmlec_data<FLOATTYPE>* data = reinterpret_cast<lmlec_data<FLOATTYPE>*>(adata);

	int        m   = mm + data->ncnstr;
	FLOATTYPE* c   = data->c;
	FLOATTYPE* Z   = data->Z;
	FLOATTYPE* p   = data->p;
	FLOATTYPE* jac = data->jac;
	/* p=c + Z*pp */
	for( int i = 0; i < m; ++i )
	{
		FLOATTYPE* Zimm = Z + i*mm;
		FLOATTYPE  sum_ = c[i];
		for( int j = 0; j < mm; ++j )
		{
			sum_ += Zimm[j] * pp[j]; // sum_ += Z[i*mm+j] * pp[j];
		}
		p[i] = sum_;
	}

	(*(data->jacf))( p, jac, m, n, data->adata );

	/* compute jac*Z in jacjac */
	if( n*m <= __BLOCKSZ__SQ ) // this is a small problem
	{						   /* This is the straightforward way to compute jac*Z. However, due to
							    * its noncontinuous memory access pattern, it incures many cache misses when
							    * applied to large minimization problems (i.e. problems involving a large
							    * number of free variables and measurements), in which jac is too large to
							    * fit in the L1 cache. For such problems, a cache-efficient blocking scheme
							    * is preferable. On the other hand, the straightforward algorithm is faster
							    * on small problems since in this case it avoids the overheads of blocking.
							    */
		for( int i = 0; i < n; ++i )
		{
			FLOATTYPE* jim   = jac    + i*m;
			FLOATTYPE* jjimm = jacjac + i*mm;
			for( int j = 0; j < mm; ++j )
			{
				FLOATTYPE sum_ = zero<FLOATTYPE>();
				for( int l = 0; l < m; ++l )
				{
					sum_ += jim[l] * Z[l*mm+j]; // sum_ += jac[i*m+l] * Z[l*mm+j];
				}
				jjimm[j] = sum_; // jacjac[i*mm+j] = sum_;
			}
		}
	}
	else // this is a large problem
	{    /* Cache efficient computation of jac*Z based on blocking
		  */
		for( int jj = 0; jj < mm; jj += __BLOCKSZ__ )
		{
			for( int i = 0; i < n; ++i )
			{
				// jacjac[i*mm+j] = 0.0;
				FLOATTYPE* jjimm = jacjac+i*mm;
				std::fill( jjimm + jj, jjimm + std::min(jj + __BLOCKSZ__, mm), zero<FLOATTYPE>() );
			}

			for( int ll = 0; ll < m; ll += __BLOCKSZ__ )
			{
				for( int i = 0; i < n; ++i )
				{
					FLOATTYPE* jjimm = jacjac + i*mm;
					FLOATTYPE* jim   = jac    + i*mm;
					for( int j = jj; j < std::min(jj + __BLOCKSZ__, mm); ++j )
					{
						FLOATTYPE sum_ = zero<FLOATTYPE>();
						for( int l = ll; l < std::min(ll + __BLOCKSZ__, m); ++l )
							sum_ += jim[l] * Z[l*mm+j]; //jac[i*m+l] * Z[l*mm+j];
						jjimm[j] += sum_;                //jacjac[i*mm+j] += sum_;
					}
				}
			}
		}
	}
}

/* 
* This function is similar to LEVMAR_DER except that the minimization
* is performed subject to the linear constraints A p=b, A is kxm, b kx1
*
* This function requires an analytic Jacobian. In case the latter is unavailable,
* use LEVMAR_LEC_DIF() bellow
*
*/
template<typename FLOATTYPE>
int levmar_lec_der_impl( FPTR<FLOATTYPE> func,    /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
						 FPTR<FLOATTYPE> jacf,    /* function to evaluate the Jacobian \part x / \part p                                */
						 FLOATTYPE*      p,       /* I/O: initial parameter estimates. On output has the estimated solution             */
						 FLOATTYPE*      x,       /* I: measurement vector. NULL implies a zero vector                                  */
						 int             m,       /* I: parameter vector dimension (i.e. #unknowns)                                     */
						 int             n,       /* I: measurement vector dimension                                                    */
						 FLOATTYPE*      A,       /* I: constraints matrix, kxm                                                         */
						 FLOATTYPE*      b,       /* I: right hand constraints vector, kx1                                              */
						 int             k,       /* I: number of constraints (i.e. A's #rows)                                          */
						 int             itmax,   /* I: maximum number of iterations                                                    */
						 FLOATTYPE       opts[4], /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
						                           * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
						                           */
						 FLOATTYPE       info[LMPP_INFO_SZ],
						                          /* O: information regarding the minimization. Set to NULL if don't care
						                           * info[0]   = ||e||_2 at initial p.
						                           * info[1-4] = [ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
						                           * info[5]   = # iterations,
						                           * info[6]   = reason for terminating: 1 - stopped by small gradient J^T e
						                           *                                     2 - stopped by small Dp
						                           *                                     3 - stopped by itmax
						                           *                                     4 - singular matrix. Restart from current p with increased mu 
						                           *                                     5 - no further error reduction is possible. Restart with increased mu
						                           *                                     6 - stopped by small ||e||_2
						                           *                                     7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
						                           * info[7]   = # function evaluations
						                           * info[8]   = # Jacobian evaluations
						                           * info[9]   = # linear systems solved, i.e. # attempts for reducing error
						                           */
						 FLOATTYPE* work,         /* working memory at least LM_LEC_DER_WORKSZ() reals large, allocated if NULL */
						 FLOATTYPE* covar,        /* O: Covariance matrix corresponding to LS solution; mxm. Set to NULL if not needed. */
						 void*      adata )       /* pointer to possibly additional data, passed uninterpreted to func & jacf.
						                           * Set to NULL if not needed
						                           */
{
	const int mm = m-k;
	if( !jacf )
	{
		std::cerr << "No function specified for computing the Jacobian in levmar_lec_der()\n";
		std::cerr << "If no such function is available, use levmar_lec_dif() rather than levmar_lec_der()\n";
		return LMPP_ERROR;
	}

	if( n < mm )
	{
		std::cerr << "levmar_lec_der(): cannot solve a problem with fewer measurements + equality constraints ["
			<< n << " + " << k << "] than unknowns [" << m << "]\n";
		return LMPP_ERROR;
	}

	std::vector<FLOATTYPE> buf;
	lmlec_data<FLOATTYPE>  data;

	buf.resize(2*m + m*mm + n*m + mm);

	FLOATTYPE* p0 = buf.data();
	data.p        = p;
	data.c        = p0 + m;
	data.Z        = data.c + m;
	data.jac      = data.Z + m*mm;
	data.ncnstr   = k;
	data.func     = func;
	data.jacf     = jacf;
	data.adata    = adata;
	FLOATTYPE* pp = data.jac+n*m;

	int ret = lmlec_elim<FLOATTYPE>( A, b, data.c, nullptr, data.Z, k, m ); // compute c, Z
	if( ret == LMPP_ERROR )
	{
		return LMPP_ERROR;
	}

	/* compute pp s.t. p = c + Z*pp or (Z^T Z)*pp=Z^T*(p-c)
	 * Due to orthogonality, Z^T Z = I and the last equation
	 * becomes pp=Z^T*(p-c). Also, save the starting p in p0
	 */
	for( int i = 0; i < m; ++i )
	{
		p0[i] = p[i];
		p[i] -= data.c[i];
	}

	/* Z^T*(p-c) */
	for( int i = 0; i < mm; ++i )
	{
		FLOATTYPE sum_ = zero<FLOATTYPE>();
		for( int j = 0; j < m; ++j )
		{
			sum_ += data.Z[j*mm+i] * p[j];
		}
		pp[i] = sum_;
	}

	/* compute the p corresponding to pp (i.e. c + Z*pp) and compare with p0 */
	for( int i = 0; i < m; ++i )
	{
		FLOATTYPE* Zimm = data.Z + i*mm;
		FLOATTYPE  sum_ = data.c[i];
		for( int j = 0; j < mm; ++j )
		{
			sum_ += Zimm[j] * pp[j]; // sum_ += Z[i*mm+j] * pp[j];
		}
		if( std::abs(sum_-p0[i]) > FLOATTYPE(1E-03) )
		{
			std::cerr << "Warning: component " << i << " of starting point not feasible in levmar_lec_der()! ["
					  << p0[i] << " reset to " << sum_ << "]\n";
		}
	}

	FLOATTYPE locinfo[LMPP_INFO_SZ];
	if( !info ) /* make sure that LEVMAR_DER() is called with non-null info */
	{
		info = locinfo;
	}
	/* note that covariance computation is not requested from LEVMAR_DER() */
	ret = levmar_der( lmlec_func<FLOATTYPE>, lmlec_jacf, pp, x, mm, n, itmax, opts, info, work, nullptr, (void *)&data );

	/* p = c + Z*pp */
	for( int i = 0; i < m; ++i )
	{
		FLOATTYPE* Zimm = data.Z + i*mm;
		FLOATTYPE  sum_ = data.c[i];
		for( int j = 0; j < mm; ++j )
		{
			sum_ += Zimm[j] * pp[j]; // sum_ += Z[i*mm+j] * pp[j];
		}
		p[i] = sum_;
	}

	/* compute the covariance from the Jacobian in data.jac */
	if( covar )
	{
		levmar_trans_mat_mat_mult( data.jac, covar, n, m ); /* covar = J^T J */
		levmar_covar( covar, covar, info[1], m, n );
	}

	return ret;
}

/* Similar to the LEVMAR_LEC_DER() function above, except that the Jacobian is approximated
 * with the aid of finite differences (forward or central, see the comment for the opts argument)
 */
template<typename FLOATTYPE>
int levmar_lec_dif_impl( FPTR<FLOATTYPE> func, /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
						 FLOATTYPE* p,         /* I/O: initial parameter estimates. On output has the estimated solution */
						 FLOATTYPE* x,         /* I: measurement vector. NULL implies a zero vector */
						 int        m,         /* I: parameter vector dimension (i.e. #unknowns) */
						 int        n,         /* I: measurement vector dimension */
						 FLOATTYPE* A,         /* I: constraints matrix, kxm */
						 FLOATTYPE* b,         /* I: right hand constraints vector, kx1 */
						 int        k,         /* I: number of constraints (i.e. A's #rows) */
						 int        itmax,     /* I: maximum number of iterations */
						 FLOATTYPE  opts[5],   /* I: opts[0-3] = minim. options [\mu, \epsilon1, \epsilon2, \epsilon3, \delta]. Respectively the
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
						 FLOATTYPE* work,      /* working memory at least LM_LEC_DIF_WORKSZ() reals large, allocated if NULL */
						 FLOATTYPE* covar,     /* O: Covariance matrix corresponding to LS solution; mxm. Set to NULL if not needed. */
						 void*      adata )   /* pointer to possibly additional data, passed uninterpreted to func.
						                       * Set to NULL if not needed
						                       */
{
	const int mm = m-k;

	if( n < mm )
	{
		std::cerr << "levmar_lec_dif(): cannot solve a problem with fewer measurements + equality constraints ["
			<< n << " + " << k << "] than unknowns [" << m << "]\n";
		return LMPP_ERROR;
	}

	std::vector<FLOATTYPE> buf;
	lmlec_data<FLOATTYPE>  data;

	buf.resize(2*m + m*mm + mm);

	FLOATTYPE* p0 = buf.data();
	data.p        = p;
	data.c        = p0 + m;
	data.Z        = data.c + m;
	data.jac      = nullptr;
	data.ncnstr   = k;
	data.func     = func;
	data.jacf     = nullptr;
	data.adata    = adata;
	FLOATTYPE* pp = data.Z + m*mm;

	int ret = lmlec_elim<FLOATTYPE>( A, b, data.c, nullptr, data.Z, k, m ); // compute c, Z
	if( ret == LMPP_ERROR )
	{
		return LMPP_ERROR;
	}

	/* compute pp s.t. p = c + Z*pp or (Z^T Z)*pp=Z^T*(p-c)
	 * Due to orthogonality, Z^T Z = I and the last equation
	 * becomes pp=Z^T*(p-c). Also, save the starting p in p0
	 */
	for( int i = 0; i < m; ++i )
	{
		p0[i] = p[i];
		p[i] -= data.c[i];
	}

	/* Z^T*(p-c) */
	for( int i = 0; i < mm; ++i )
	{
		FLOATTYPE sum_ = zero<FLOATTYPE>();
		for( int j = 0; j < m; ++j )
		{
			sum_ += data.Z[j*mm+i] * p[j];
		}
		pp[i] = sum_;
	}

	/* compute the p corresponding to pp (i.e. c + Z*pp) and compare with p0 */
	for( int i = 0; i < m; ++i )
	{
		FLOATTYPE* Zimm = data.Z + i*mm;
		FLOATTYPE  sum_ = data.c[i];
		for( int j = 0; j < mm; ++j )
		{
			sum_ += Zimm[j] * pp[j]; // sum_ += Z[i*mm+j] * pp[j];
		}

		if( std::abs(sum_-p0[i]) > FLOATTYPE(1E-03) )
		{
			std::cerr << "Warning: component " << i << " of starting point not feasible in levmar_lec_dif()! ["
				<< p0[i] << " reset to " << sum_ << "]\n";
		}
	}

	FLOATTYPE locinfo[LMPP_INFO_SZ];
	if( !info ) /* make sure that LEVMAR_DER() is called with non-null info */
	{
		info = locinfo;
	}

	/* note that covariance computation is not requested from LEVMAR_DIF() */
	ret = levmar_dif( lmlec_func<FLOATTYPE>, pp, x, mm, n, itmax, opts, info, work, NULL, (void *)&data);

	/* p=c + Z*pp */
	for( int i = 0; i < m; ++i )
	{
		FLOATTYPE* Zimm = data.Z + i*mm;
		FLOATTYPE  sum_ = data.c[i];
		for( int j = 0; j < mm; ++j )
		{
			sum_ += Zimm[j] * pp[j]; // sum_ += Z[i*mm+j] * pp[j];
		}
		p[i] = sum_;
	}

	/* compute the Jacobian with finite differences and use it to estimate the covariance */
	if( covar )
	{
		std::vector<FLOATTYPE> hx_v;
		hx_v.resize(2*n+n*m);

		FLOATTYPE* hx  = hx_v.data();
		FLOATTYPE* wrk = hx  + n;
		FLOATTYPE* jac = wrk + n;

		(*func)( p, hx, m, n, adata ); /* evaluate function at p */
		levmar_fdif_forw_jac_approx( func, p, hx, wrk, FLOATTYPE(LMPP_DIFF_DELTA), jac, m, n, adata ); /* compute the Jacobian at p */
		levmar_trans_mat_mat_mult  ( jac, covar, n, m ); /* covar = J^T J */
		levmar_covar               ( covar, covar, info[1], m, n );
	}

	return ret;
}

#ifdef LMPP_DBL_PREC
/* linear equation constrained minimization */
int levmar_lec_der( FPTR<double> func,
					FPTR<double> jacf,
					double*      p,
					double*      x,
					int          m,
					int          n,
					double*      A,
					double*      b,
					int          k,
					int          itmax,
					double*      opts,
					double*      info,
					double*      work,
					double*      covar,
					void*        adata )
{
	return levmar_lec_der_impl<double>( func, jacf, p, x, m, n, A, b, k, itmax, opts, info, work, covar, adata );
}

int levmar_lec_dif( FPTR<double> func,
					double*      p,
					double*      x,
					int          m,
					int          n,
					double*      A,
					double*      b,
					int          k,
					int          itmax,
					double*      opts,
					double*      info,
					double*      work,
					double*      covar,
					void*        adata )
{
	return levmar_lec_dif_impl<double>( func, p, x, m, n, A, b, k, itmax, opts, info, work, covar, adata );
}
#endif /* LMPP_DBL_PREC */

#ifdef LMPP_SNGL_PREC
/* linear equation constrained minimization */
int levmar_lec_der( FPTR<float> func,
					FPTR<float> jacf,
					float*      p,
					float*      x,
					int         m,
					int         n,
					float*      A,
					float*      b,
					int         k,
					int         itmax,
					float*      opts,
					float*      info,
					float*      work,
					float*      covar,
					void*       adata )
{
	return levmar_lec_der_impl<float>( func, jacf, p, x, m, n, A, b, k, itmax, opts, info, work, covar, adata );
}

int levmar_lec_dif( FPTR<float> func,
					float*      p,
					float*      x,
					int         m,
					int         n,
					float*      A,
					float*      b,
					int         k,
					int         itmax,
					float*      opts,
					float*      info,
					float*      work,
					float*      covar,
					void*       adata )
{
	return levmar_lec_dif_impl<float>( func, p, x, m, n, A, b, k, itmax, opts, info, work, covar, adata );
}
#endif /* LMPP_SNGL_PREC */

#endif /* HAVE_LAPACK */

