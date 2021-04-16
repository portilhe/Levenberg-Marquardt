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
	/* 
	////////////////////////////////////////////////////////////////////////////////////
	// 
	//  Prototypes and definitions for the Levenberg - Marquardt minimization algorithm
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
	////////////////////////////////////////////////////////////////////////////////////
	*/

#ifndef _LEVMAR_H_
#define _LEVMAR_H_

#include "config_lmpp.h"

template<typename FLOATTYPE>
using FPTR = void (*)( FLOATTYPE* p, FLOATTYPE* hx, int m, int n, void* adata );

#ifdef LMPP_DBL_PREC
/* LM, with & without Jacobian */
/* unconstrained minimization */
LEVMAR_LIB_EXPORT
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
				void*        adata );

LEVMAR_LIB_EXPORT
int levmar_dif( FPTR<double> func,
				double*      p,
				double*      x,
				int             m,
				int             n,
				int             itmax,
				double*      opts,
				double*      info,
				double*      work,
				double*      covar,
				void*           adata );

/* box-constrained minimization */
LEVMAR_LIB_EXPORT
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
				   double*      opts,
				   double*      info,
				   double*      work,
				   double*      covar,
				   void*        adata );

LEVMAR_LIB_EXPORT
int levmar_bc_dif( FPTR<double> func,
				   double*      p,
				   double*      x,
				   int          m,
				   int          n,
				   double*      lb,
				   double*      ub,
				   double*      dscl,
				   int          itmax,
				   double*      opts,
				   double*      info,
				   double*      work,
				   double*      covar,
				   void*        adata );

/* linear equation constrained minimization */
LEVMAR_LIB_EXPORT
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
					void*        adata );

LEVMAR_LIB_EXPORT
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
					void*        adata );

/* box & linear equation constrained minimization */
LEVMAR_LIB_EXPORT
int levmar_blec_der( FPTR<double> func,
					 FPTR<double> jacf,
					 double*      p,
					 double*      x,
					 int          m,
					 int          n,
					 double*      lb,
					 double*      ub,
					 double*      A,
					 double*      b,
					 int          k,
					 double*      wghts,
					 int          itmax,
					 double*      opts,
					 double*      info,
					 double*      work,
					 double*      covar,
					 void*        adata );

LEVMAR_LIB_EXPORT
int levmar_blec_dif( FPTR<double> func,
					 double*      p,
					 double*      x,
					 int          m,
					 int          n,
					 double*      lb,
					 double*      ub,
					 double*      A,
					 double*      b,
					 int          k,
					 double*      wghts,
					 int          itmax,
					 double*      opts,
					 double*      info,
					 double*      work,
					 double*      covar,
					 void*        adata );

/* box, linear equations & inequalities constrained minimization */
LEVMAR_LIB_EXPORT
int levmar_bleic_der( FPTR<double> func,
					  FPTR<double> jacf,
					  double*      p,
					  double*      x,
					  int          m,
					  int          n,
					  double*      lb,
					  double*      ub,
					  double*      A,
					  double*      b,
					  int          k1,
					  double*      C,
					  double*      d,
					  int          k2,
					  int          itmax,
					  double*      opts,
					  double*      info,
					  double*      work,
					  double*      covar,
					  void*        adata );

LEVMAR_LIB_EXPORT
int levmar_bleic_dif( FPTR<double> func,
					  double*      p,
					  double*      x,
					  int          m,
					  int          n,
					  double*      lb,
					  double*      ub,
					  double*      A,
					  double*      b,
					  int          k1,
					  double*      C,
					  double*      d,
					  int          k2,
					  int          itmax,
					  double*      opts,
					  double*      info,
					  double*      work,
					  double*      covar,
					  void*        adata );

/* box & linear inequality constraints */
LEVMAR_LIB_EXPORT
int levmar_blic_der( FPTR<double> func,
					 FPTR<double> jacf,
					 double*      p,
					 double*      x,
					 int          m,
					 int          n,
					 double*      lb,
					 double*      ub,
					 double*      C,
					 double*      d,
					 int          k2,
					 int          itmax,
					 double       opts[4],
					 double       info[LMPP_INFO_SZ],
					 double*      work,
					 double*      covar,
					 void*        adata );

LEVMAR_LIB_EXPORT
int levmar_blic_dif( FPTR<double> func,
					 double*      p,
					 double*      x,
					 int          m,
					 int          n,
					 double*      lb,
					 double*      ub,
					 double*      C,
					 double*      d,
					 int          k2,
					 int          itmax,
					 double       opts[5],
					 double       info[LMPP_INFO_SZ],
					 double*      work,
					 double*      covar,
					 void*        adata );

/* linear equation & inequality constraints */
LEVMAR_LIB_EXPORT
int levmar_leic_der( FPTR<double> func,
					 FPTR<double> jacf,
					 double*      p,
					 double*      x,
					 int          m,
					 int          n,
					 double*      A,
					 double*      b,
					 int          k1,
					 double*      C,
					 double*      d,
					 int          k2,
					 int          itmax,
					 double       opts[4],
					 double       info[LMPP_INFO_SZ],
					 double*      work,
					 double*      covar,
					 void*        adata );

LEVMAR_LIB_EXPORT
int levmar_leic_dif( FPTR<double> func,
					 double*      p,
					 double*      x,
					 int          m,
					 int          n,
					 double*      A,
					 double*      b,
					 int          k1,
					 double*      C,
					 double*      d,
					 int          k2,
					 int          itmax,
					 double       opts[5],
					 double       info[LMPP_INFO_SZ],
					 double*      work,
					 double*      covar,
					 void*        adata );

/* linear inequality constraints */
LEVMAR_LIB_EXPORT
int levmar_lic_der( FPTR<double> func,
					FPTR<double> jacf,
					double*      p,
					double*      x,
					int          m,
					int          n,
					double*      C,
					double*      d,
					int          k2,
					int          itmax,
					double       opts[4],
					double       info[LMPP_INFO_SZ],
					double*      work,
					double*      covar,
					void*        adata );

LEVMAR_LIB_EXPORT
int levmar_lic_dif( FPTR<double> func,
					double*      p,
					double*      x,
					int          m,
					int          n,
					double*      C,
					double*      d,
					int          k2,
					int          itmax,
					double       opts[5],
					double       info[LMPP_INFO_SZ],
					double*      work,
					double*      covar,
					void*        adata );

/* Jacobian verification */
LEVMAR_LIB_EXPORT
void levmar_chkjac( FPTR<double> func, FPTR<double> jacf, double* p, int m, int n, void* adata, double* err );

/* miscellaneous: standard deviation, coefficient of determination (R2),
 *                Pearson's correlation coefficient for best-fit parameters
 */
LEVMAR_LIB_EXPORT double levmar_stddev( double* covar, int m, int i );
LEVMAR_LIB_EXPORT double levmar_corcoef( double* covar , int m, int i, int j );
LEVMAR_LIB_EXPORT double levmar_R2( FPTR<double> func, double* p, double* x, int m, int n, void* adata );
#endif // LMPP_DBL_PREC

#ifdef LMPP_SNGL_PREC
/* LM, with & without Jacobian */
/* unconstrained minimization */
LEVMAR_LIB_EXPORT
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
				void*       adata );

LEVMAR_LIB_EXPORT
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
				void*       adata );

/* box-constrained minimization */
LEVMAR_LIB_EXPORT
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
				   float*      opts,
				   float*      info,
				   float*      work,
				   float*      covar,
				   void*       adata );

LEVMAR_LIB_EXPORT
int levmar_bc_dif( FPTR<float> func,
				   float*      p,
				   float*      x,
				   int         m,
				   int         n,
				   float*      lb,
				   float*      ub,
				   float*      dscl,
				   int         itmax,
				   float*      opts,
				   float*      info,
				   float*      work,
				   float*      covar,
				   void*       adata );

/* linear equation constrained minimization */
LEVMAR_LIB_EXPORT
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
					void*       adata );

LEVMAR_LIB_EXPORT
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
					void*       adata );

/* box & linear equation constrained minimization */
LEVMAR_LIB_EXPORT
int levmar_blec_der( FPTR<float> func,
					 FPTR<float> jacf,
					 float*      p,
					 float*      x,
					 int         m,
					 int         n,
					 float*      lb,
					 float*      ub,
					 float*      A,
					 float*      b,
					 int         k,
					 float*      wghts,
					 int         itmax,
					 float*      opts,
					 float*      info,
					 float*      work,
					 float*      covar,
					 void*       adata );

LEVMAR_LIB_EXPORT
int levmar_blec_dif( FPTR<float> func,
					 float*      p,
					 float*      x,
					 int         m,
					 int         n,
					 float*      lb,
					 float*      ub,
					 float*      A,
					 float*      b,
					 int         k,
					 float*      wghts,
					 int         itmax,
					 float*      opts,
					 float*      info,
					 float*      work,
					 float*      covar,
					 void*       adata );

/* box, linear equations & inequalities constrained minimization */
LEVMAR_LIB_EXPORT
int levmar_bleic_der( FPTR<float> func,
					  FPTR<float> jacf,
					  float*      p,
					  float*      x,
					  int         m,
					  int         n,
					  float*      lb,
					  float*      ub,
					  float*      A,
					  float*      b,
					  int         k1,
					  float*      C,
					  float*      d,
					  int         k2,
					  int         itmax,
					  float*      opts,
					  float*      info,
					  float*      work,
					  float*      covar,
					  void*       adata );

LEVMAR_LIB_EXPORT
int levmar_bleic_dif( FPTR<float> func,
					  float*      p,
					  float*      x,
					  int         m,
					  int         n,
					  float*      lb,
					  float*      ub,
					  float*      A,
					  float*      b,
					  int         k1,
					  float*      C,
					  float*      d,
					  int         k2,
					  int         itmax,
					  float*      opts,
					  float*      info,
					  float*      work,
					  float*      covar,
					  void*       adata );

/* box & linear inequality constraints */
LEVMAR_LIB_EXPORT
int levmar_blic_der( FPTR<float> func,
					 FPTR<float> jacf,
					 float*      p,
					 float*      x,
					 int         m,
					 int         n,
					 float*      lb,
					 float*      ub,
					 float*      C,
					 float*      d,
					 int         k2,
					 int         itmax,
					 float       opts[4],
					 float       info[LMPP_INFO_SZ],
					 float*      work,
					 float*      covar,
					 void*       adata );

LEVMAR_LIB_EXPORT
int levmar_blic_dif( FPTR<float> func,
					 float*      p,
					 float*      x,
					 int         m,
					 int         n,
					 float*      lb,
					 float*      ub,
					 float*      C,
					 float*      d,
					 int         k2,
					 int         itmax,
					 float       opts[5],
					 float       info[LMPP_INFO_SZ],
					 float*      work,
					 float*      covar,
					 void*       adata );

/* linear equation & inequality constraints */
LEVMAR_LIB_EXPORT
int levmar_leic_der( FPTR<float> func,
					 FPTR<float> jacf,
					 float*      p,
					 float*      x,
					 int         m,
					 int         n,
					 float*      A,
					 float*      b,
					 int         k1,
					 float*      C,
					 float*      d,
					 int         k2,
					 int         itmax,
					 float       opts[4],
					 float       info[LMPP_INFO_SZ],
					 float*      work,
					 float*      covar,
					 void*       adata );

LEVMAR_LIB_EXPORT
int levmar_leic_dif( FPTR<float> func,
					 float*      p,
					 float*      x,
					 int         m,
					 int         n,
					 float*      A,
					 float*      b,
					 int         k1,
					 float*      C,
					 float*      d,
					 int         k2,
					 int         itmax,
					 float       opts[5],
					 float       info[LMPP_INFO_SZ],
					 float*      work,
					 float*      covar,
					 void*       adata );

/* linear inequality constraints */
LEVMAR_LIB_EXPORT
int levmar_lic_der( FPTR<float> func,
					FPTR<float> jacf,
					float*      p,
					float*      x,
					int         m,
					int         n,
					float*      C,
					float*      d,
					int         k2,
					int         itmax,
					float       opts[4],
					float       info[LMPP_INFO_SZ],
					float*      work,
					float*      covar,
					void*       adata );

LEVMAR_LIB_EXPORT
int levmar_lic_dif( FPTR<float> func,
					float*      p,
					float*      x,
					int         m,
					int         n,
					float*      C,
					float*      d,
					int         k2,
					int         itmax,
					float       opts[5],
					float       info[LMPP_INFO_SZ],
					float*      work,
					float*      covar,
					void*       adata );

/* Jacobian verification */
LEVMAR_LIB_EXPORT
void levmar_chkjac( FPTR<float> func, FPTR<float> jacf, float* p, int m, int n, void* adata, float* err );

/* miscellaneous: standard deviation, coefficient of determination (R2),
 *                Pearson's correlation coefficient for best-fit parameters
 */
LEVMAR_LIB_EXPORT float levmar_stddev( float* covar, int m, int i );
LEVMAR_LIB_EXPORT float levmar_corcoef( float* covar , int m, int i, int j );
LEVMAR_LIB_EXPORT float levmar_R2( FPTR<float> func, float* p, float* x, int m, int n, void* adata );
#endif // LMPP_SNGL_PREC

#ifdef LMPP_HAVE_LAPACK

#endif /* LMPP_HAVE_LAPACK */

#ifdef LMPP_SNGL_PREC
void levmar_locscale( FPTR<float> func,
					  float *p,
					  float *x,
					  int    m,
					  int    n,
					  void  *adata,
					  int    howto,
					  float  locscl[2],
					  float  **residptr );

int levmar_outlid( float *r, int n, float thresh, float ls[2], char *outlmap );
#endif /* LMPP_SNGL_PREC */

#endif /* _LEVMAR_H_ */
