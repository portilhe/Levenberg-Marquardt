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
	//  Demonstration driver program for the Levenberg - Marquardt minimization
	//  algorithm
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
 * Levenberg-Marquardt minimization demo driver. Only the double precision versions
 * are tested here. See the Meyer case for an example of verifying the Jacobian 
 ********************************************************************************/

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <ostream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

#include "levmar.h"

#ifndef LMPP_DBL_PREC
#error Demo program assumes that levmar has been compiled with double precision, see LM_DBL_PREC!
#endif


/* Sample functions to be minimized with LM and their Jacobians.
 * More test functions at http://www.csit.fsu.edu/~burkardt/f_src/test_nls/test_nls.html
 * Check also the CUTE problems collection at ftp://ftp.numerical.rl.ac.uk/pub/cute/;
 * CUTE is searchable through http://numawww.mathematik.tu-darmstadt.de:8081/opti/select.html
 * CUTE problems can also be solved through the AMPL web interface at http://www.ampl.com/TRYAMPL/startup.html
 *
 * Nonlinear optimization models in AMPL can be found at http://www.princeton.edu/~rvdb/ampl/nlmodels/
 */

template<typename T>
std::remove_cv_t<std::remove_reference_t<T>> sq(T const&& x)
{
	return x*x;
}

template<typename T>
std::remove_cv_t<std::remove_reference_t<T>> sq(T&& x)
{
	return x*x;
}

typedef int (*tfunc_ptr)( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag );


/* Rosenbrock function, global minimum at (1, 1) */
#define ROSD 105.0
void ros( double *p, double *x, int m, int n, void *data )
{
	for( int i = 0; i < n; ++i )
	{
		x[i] = ( sq(1.0 - p[0]) + ROSD * sq(p[1] - sq(p[0])) );
	}
}

void jacros( double *p, double *jac, int m, int n, void *data )
{
	for( int i = 0; i < 2*n; i += 2 )
	{
		jac[i  ] = ( -2 + 2 * p[0] - 4 * ROSD * (p[1] - sq(p[0])) * p[0]);
		jac[i+1] = ( 2 * ROSD * (p[1] - sq(p[0])) );
	}
}


#define MODROSLAM 1E02
/* Modified Rosenbrock problem, global minimum at (1, 1) */
void modros( double *p, double *x, int m, int n, void *data )
{
	for( int i = 0; i < n; i +=3 )
	{
		x[i  ] = 10. * (p[1] - sq(p[0]));
		x[i+1] = 1.0 - p[0];
		x[i+2] = MODROSLAM;
	}
}

void jacmodros( double *p, double *jac, int m, int n, void *data )
{
	for( int i = 0; i < 2*n; i += 6 )
	{
		jac[i  ] = -20.0 * p[0];
		jac[i+1] = 10.0;

		jac[i+2] = -1.0;
		jac[i+3] = 0.0;

		jac[i+4] = 0.0;
		jac[i+5] = 0.0;
	}
}


/* Powell's function, minimum at (0, 0) */
void powell( double *p, double *x, int m, int n, void *data )
{
	for( int i = 0; i < n; i +=2 )
	{
		x[i  ] = p[0];
		x[i+1] = 10.0 * p[0] / (p[0]+0.1) + 2*sq(p[1]);
	}
}

void jacpowell( double *p, double *jac, int m, int n, void *data )
{
	for( int i = 0; i < 2*n; i += 4 )
	{
		jac[i  ] = 1.0;
		jac[i+1] = 0.0;

		jac[i+2] = 1.0 / sq(p[0] + 0.1);
		jac[i+3] = 4.0 * p[1];
	}
}

/* Wood's function, minimum at (1, 1, 1, 1) */
void wood( double *p, double *x, int m, int n, void *data )
{
	for( int i = 0; i < n; i += 6 )
	{
		x[i  ] = 10.0 * (p[1] - sq(p[0]));
		x[i+1] = 1.0 - p[0];
		x[i+2] = std::sqrt(90.0) * (p[3] - sq(p[2]));
		x[i+3] = 1.0 - p[2];
		x[i+4] = std::sqrt(10.0) * (p[1]+p[3] - 2.0);
		x[i+5] = (p[1] - p[3]) / std::sqrt(10.0);
	}
}

/* Meyer's (reformulated) problem, minimum at (2.48, 6.18, 3.45) */
void meyer( double *p, double *x, int m, int n, void *data )
{
	for( int i = 0; i < n; ++i )
	{
		double ui = 0.45 + 0.05 * i;
		x[i]      = p[0] * exp( 10.0 * p[1]/ (ui+p[2]) - 13.0 );
	}
}

void jacmeyer( double *p, double *jac, int m, int n, void *data )
{
	for( int i = 0; i < 3*n; i += 3 )
	{
		double ui = 0.45 + 0.05 * (i/3);
		double aux1 = ui + p[2];
		double aux2 = exp( 10.0 * p[1] / aux1 - 13.0 );

		jac[i  ] =  aux2;
		jac[i+1] =  10.0 * p[0] * aux2 / aux1;
		jac[i+2] = -10.0 * p[0] * p[1] * aux2 / sq(aux1);
	}
}

/* Osborne's problem, minimum at (0.3754, 1.9358, -1.4647, 0.0129, 0.0221) */
void osborne( double *p, double *x, int m, int n, void *data )
{
	for( int i = 0; i < n; ++i )
	{
		double t = 10.*i;
		x[i] = p[0] + p[1] * exp(-p[3]*t) + p[2] * exp(-p[4]*t);
	}
}

void jacosborne( double *p, double *jac, int m, int n, void *data )
{
	for( int i = 0; i < n; i += 5 )
	{
		double t = 2. * i;
		double aux1 = exp(-p[3]*t);
		double aux2 = exp(-p[4]*t);

		jac[i  ] = 1.0;
		jac[i+1] = aux1;
		jac[i+2] = aux2;
		jac[i+3] = -p[1] * t * aux1;
		jac[i+4] = -p[2] * t * aux2;
	}
}

//#ifndef M_PI
//#define M_PI   3.14159265358979323846  /* pi */
//#endif
/* helical valley function, minimum at (1.0, 0.0, 0.0) */
void helval( double *p, double *x, int m, int n, void *data )
{
	double theta;
	if( p[0] < 0.0 )
	{
		theta = std::atan( p[1] / p[0] ) / (2.0 * M_PI) + 0.5;
	}
	else if( 0.0 < p[0] )
	{
		theta = std::atan( p[1] / p[0]) / (2.0 * M_PI);
	}
	else
	{
		theta = p[1] >=0 ? 0.25 : -0.25;
	}

	x[0] = 10.0 * ( p[2] - 10.0 * theta );
	x[1] = 10.0 * ( sqrt(sq(p[0]) + sq(p[1])) - 1.0 );
	x[2] = p[2];
}

void jachelval( double *p, double *jac, int m, int n, void *data )
{
	double aux = sq(p[0]) + sq(p[1]);

	jac[0] = 50.0 * p[1] / (M_PI * aux);
	jac[1] =-50.0 * p[0] / (M_PI * aux);
	jac[2] = 10.0;

	jac[3] = 10.0 * p[0] / std::sqrt(aux);
	jac[4] = 10.0 * p[1] / std::sqrt(aux);
	jac[5] = 0.0;

	jac[6] = 0.0;
	jac[7] = 0.0;
	jac[8] = 1.0;
}

/* Boggs - Tolle problem 3 (linearly constrained), minimum at (-0.76744, 0.25581, 0.62791, -0.11628, 0.25581)
 * constr1: p[0] + 3*p[1] = 0;
 * constr2: p[2] + p[3] - 2*p[4] = 0;
 * constr3: p[1] - p[4] = 0;
 */
void bt3( double *p, double *x, int m, int n, void *data )
{
	double t1 = sq( p[0] - p[1] );
	double t2 = sq( p[1] + p[2] - 2.0 );
	double t3 = sq( p[3] - 1.0 );
	double t4 = sq( p[4] - 1.0 );

	for(int i = 0; i < n; ++i )
	    x[i] = t1 + t2 + t3 + t4;
}

void jacbt3( double *p, double *jac, int m, int n, void *data )
{
	double t1 = p[0] - p[1];
	double t2 = p[1] + p[2] - 2.0;
	double t3 = p[3] - 1.0;
	double t4 = p[4] - 1.0;

	for( int i = 0; i < 5*n; i += 5 )
	{
		jac[i  ] = 2.0 * t1;
		jac[i+1] = 2.0 * (t2-t1);
		jac[i+2] = 2.0 * t2;
		jac[i+3] = 2.0 * t3;
		jac[i+4] = 2.0 * t4;
	}
}

/* Hock - Schittkowski problem 28 (linearly constrained), minimum at (0.5, -0.5, 0.5)
 * constr1: p[0] + 2*p[1] + 3*p[2] = 1;
 */
void hs28( double *p, double *x, int m, int n, void *data )
{
	double t1 = sq(p[0] + p[1]);
	double t2 = sq(p[1] + p[2]);

	for( int i = 0; i < n; ++i )
		x[i] = t1 + t2;
}

void jachs28( double *p, double *jac, int m, int n, void *data )
{
	double t1 = p[0] + p[1];
	double t2 = p[1] + p[2];

	for( int i = 0; i < 3*n; i += 3 )
	{
		jac[i  ] = 2.0 * t1;
		jac[i+1] = 2.0 * (t1 + t2);
		jac[i+2] = 2.0 * t2;
	}
}

/* Hock - Schittkowski problem 48 (linearly constrained), minimum at (1.0, 1.0, 1.0, 1.0, 1.0)
 * constr1: sum {i in 0..4} p[i] = 5;
 * constr2: p[2] - 2*(p[3]+p[4]) = -3;
 */
void hs48( double *p, double *x, int m, int n, void *data )
{
	double t1 = sq(p[0] -  1.0);
	double t2 = sq(p[1] - p[2]);
	double t3 = sq(p[3] - p[4]);

	for( int i = 0; i < n; ++i )
		x[i] = t1 + t2 + t3;
}

void jachs48(double *p, double *jac, int m, int n, void *data)
{
	double t1 = p[0] - 1.0;
	double t2 = p[1] - p[2];
	double t3 = p[3] - p[4];

	for( int i = 0; i < n; i += 5 )
	{
		jac[i  ] =  2.0 * t1;
		jac[i+1] =  2.0 * t2;
		jac[i+2] = -2.0 * t2;
		jac[i+3] =  2.0 * t3;
		jac[i+4] = -2.0 * t3;
	}
}

/* Hock - Schittkowski problem 51 (linearly constrained), minimum at (1.0, 1.0, 1.0, 1.0, 1.0)
 * constr1: p[0] + 3*p[1] = 4;
 * constr2: p[2] + p[3] - 2*p[4] = 0;
 * constr3: p[1] - p[4] = 0;
 */
void hs51( double *p, double *x, int m, int n, void *data )
{
	double t1 = sq( p[0] - p[1] );
	double t2 = sq( p[1] + p[2] - 2.0 );
	double t3 = sq( p[3] - 1.0 );
	double t4 = sq( p[4] - 1.0 );

	for( int i = 0; i < n; ++i )
		x[i] = t1 + t2 + t3 + t4;
}

void jachs51( double *p, double *jac, int m, int n, void *data )
{
	double t1 = p[0] - p[1];
	double t2 = p[1] + p[2] - 2.0;
	double t3 = p[3] - 1.0;
	double t4 = p[4] - 1.0;

	for( int i = 0; i < 5*n; i += 5 )
	{
		jac[i  ] = 2.0 * t1;
		jac[i+1] = 2.0 * (t2-t1);
		jac[i+2] = 2.0 * t2;
		jac[i+3] = 2.0 * t3;
		jac[i+4] = 2.0 * t4;
	}
}

/* Hock - Schittkowski problem 01 (box constrained), minimum at (1.0, 1.0)
 * constr1: p[1]>=-1.5;
 */
void hs01( double *p, double *x, int m, int n, void *data )
{
	x[0]     = 10.0 * (p[1] - sq(p[0]));
	x[1]     =  1.0 - p[0];
}

void jachs01( double *p, double *jac, int m, int n, void *data )
{
	jac[0] = -20.0 * p[0];
	jac[1] =  10.0;

	jac[2] = -1.0;
	jac[3] =  0.0;
}

/* Hock - Schittkowski MODIFIED problem 21 (box constrained), minimum at (2.0, 0.0)
 * constr1: 2 <= p[0] <=50;
 * constr2: -50 <= p[1] <=50;
 *
 * Original HS21 has the additional constraint 10*p[0] - p[1] >= 10; which is inactive
 * at the solution, so it is dropped here.
 */
void hs21( double *p, double *x, int m, int n, void *data )
{
	x[0] = p[0] / 10.0;
	x[1] = p[1];
}

void jachs21( double *p, double *jac, int m, int n, void *data )
{
	jac[0] = 0.1;
	jac[1] = 0.0;

	jac[2] = 0.0;
	jac[3] = 1.0;
}

/* Problem hatfldb (box constrained), minimum at (0.947214, 0.8, 0.64, 0.4096)
 * constri: p[i]>=0.0; (i=1..4)
 * constr5: p[1]<=0.8;
 */
void hatfldb(  double *p, double *x, int m, int n, void *data )
{
	x[0] = p[0] - 1.0;
	for( int i = 1; i < m; ++i )
		x[i] = p[i-1] - std::sqrt(p[i]);
}

void jachatfldb( double *p, double *jac, int m, int n, void *data )
{
	jac[ 0] = 1.0;
	jac[ 1] = 0.0;
	jac[ 2] = 0.0;
	jac[ 3] = 0.0;

	jac[ 4] = 1.0;
	jac[ 5] = -0.5 / std::sqrt(p[1]);
	jac[ 6] = 0.0;
	jac[ 7] = 0.0;

	jac[ 8] = 0.0;
	jac[ 9] = 1.0;
	jac[10] = -0.5 / std::sqrt(p[2]);
	jac[11] = 0.0;

	jac[12] = 0.0;
	jac[13] = 0.0;
	jac[14] = 1.0;
	jac[15] = -0.5 / std::sqrt(p[3]);
}

/* Problem hatfldc (box constrained), minimum at (1.0, 1.0, 1.0, 1.0)
 * constri: p[i]>=0.0; (i=1..4)
 * constri+4: p[i]<=10.0; (i=1..4)
 */
void hatfldc( double *p, double *x, int m, int n, void *data )
{
	x[0] = p[0] - 1.0;
	for( int i = 1; i < m-1; ++i )
	{
		x[i] = p[i-1] - std::sqrt(p[i]);
	}
	x[m-1] = p[m-1] - 1.0;
}

void jachatfldc( double *p, double *jac, int m, int n, void *data )
{
	jac[ 0] = 1.0;
	jac[ 1] = 0.0;
	jac[ 2] = 0.0;
	jac[ 3] = 0.0;

	jac[ 4] =  1.0;
	jac[ 5] = -0.5 / std::sqrt(p[1]);
	jac[ 6] =  0.0;
	jac[ 7] =  0.0;

	jac[ 8] =  0.0;
	jac[ 9] =  1.0;
	jac[10] = -0.5 / std::sqrt(p[2]);
	jac[11] =  0.0;

	jac[12] = 0.0;
	jac[13] = 0.0;
	jac[14] = 0.0;
	jac[15] = 1.0;
}

/* Hock - Schittkowski (modified #1) problem 52 (box/linearly constrained), minimum at (-0.09, 0.03, 0.25, -0.19, 0.03)
 * constr1: p[0] + 3*p[1] = 0;
 * constr2: p[2] +   p[3] - 2*p[4] = 0;
 * constr3: p[1] -   p[4] = 0;
 *
 * To the above 3 constraints, we add the following 5:
 * constr4: -0.09 <= p[0];
 * constr5:   0.0 <= p[1] <= 0.3;
 * constr6:          p[2] <= 0.25;
 * constr7:  -0.2 <= p[3] <= 0.3;
 * constr8:   0.0 <= p[4] <= 0.3;
 *
 */
void mod1hs52( double *p, double *x, int m, int n, void *data )
{
	x[0] =  4.0 * p[0] - p[1];
	x[1] = p[1] + p[2] - 2.0;
	x[2] = p[3] - 1.0;
	x[3] = p[4] - 1.0;
}

void jacmod1hs52( double *p, double *jac, int m, int n, void *data )
{
	jac[ 0] =  4.0;
	jac[ 1] = -1.0;
	jac[ 2] =  0.0;
	jac[ 3] =  0.0;
	jac[ 4] =  0.0;

	jac[ 5] = 0.0;
	jac[ 6] = 1.0;
	jac[ 7] = 1.0;
	jac[ 8] = 0.0;
	jac[ 9] = 0.0;

	jac[10] = 0.0;
	jac[11] = 0.0;
	jac[12] = 0.0;
	jac[13] = 1.0;
	jac[14] = 0.0;

	jac[15] = 0.0;
	jac[16] = 0.0;
	jac[17] = 0.0;
	jac[18] = 0.0;
	jac[19] = 1.0;
}


/* Hock - Schittkowski (modified #2) problem 52 (linear inequality constrained), minimum at (0.5, 2.0, 0.0, 1.0, 1.0)
 * A fifth term [(p[0]-0.5)^2] is added to the objective function and 
 * the equality contraints are replaced by the following inequalities:
 * constr1: p[0] + 3*p[1] >= -1.0;
 * constr2: p[2] +   p[3] - 2*p[4] >= -2.0;
 * constr3: p[1] -   p[4] <= 7.0;
 *
 *
 */
void mod2hs52( double *p, double *x, int m, int n, void *data )
{
	x[0] = 4.0 * p[0] - p[1];
	x[1] = p[1] + p[2] - 2.0;
	x[2] = p[3] - 1.0;
	x[3] = p[4] - 1.0;
	x[4] = p[0] - 0.5;
}

void jacmod2hs52( double *p, double *jac, int m, int n, void *data )
{
	jac[ 0] =  4.0;
	jac[ 1] = -1.0;
	jac[ 2] =  0.0;
	jac[ 3] =  0.0;
	jac[ 4] =  0.0;

	jac[ 5] = 0.0;
	jac[ 6] = 1.0;
	jac[ 7] = 1.0;
	jac[ 8] = 0.0;
	jac[ 9] = 0.0;

	jac[10] = 0.0;
	jac[11] = 0.0;
	jac[12] = 0.0;
	jac[13] = 1.0;
	jac[14] = 0.0;

	jac[15] = 0.0;
	jac[16] = 0.0;
	jac[17] = 0.0;
	jac[18] = 0.0;
	jac[19] = 1.0;

	jac[20] = 1.0;
	jac[21] = 0.0;
	jac[22] = 0.0;
	jac[23] = 0.0;
	jac[24] = 0.0;
}

/* Schittkowski (modified) problem 235 (box/linearly constrained), minimum at (-1.725, 2.9, 0.725)
 * constr1: p[0] + p[2] = -1.0;
 *
 * To the above constraint, we add the following 2:
 * constr2: p[1] - 4*p[2] = 0;
 * constr3: 0.1 <= p[1] <= 2.9;
 * constr4: 0.7 <= p[2];
 *
 */
void mods235( double *p, double *x, int m, int n, void *data )
{
	x[0] = 0.1 * (p[0] - 1.0);
	x[1] = p[1] - sq(p[0]);
}

void jacmods235( double *p, double *jac, int m, int n, void *data )
{
	jac[0] = 0.1;
	jac[1] = 0.0;
	jac[2] = 0.0;

	jac[3] = -2.0 * p[0];
	jac[4] =  1.0;
	jac[5] =  0.0;
}

/* Boggs and Tolle modified problem 7 (box/linearly constrained), minimum at (0.7, 0.49, 0.19, 1.19, -0.2)
 * We keep the original objective function & starting point and use the following constraints:
 *
 * subject to cons1:
 *  x[1] + x[2] - x[3] = 1.0;
 * subject to cons2:
 *   x[2] - x[4] + x[1] = 0.0;
 * subject to cons3:
 *   x[5] + x[1] = 0.5;
 * subject to cons4:
 *   x[5 ]>= -0.3;
 * subject to cons5:
 *    x[1] <= 0.7;
 *
 */
void modbt7( double *p, double *x, int m, int n, void *data )
{
	for( int i = 0; i < n; ++i )
		x[i] = 100.0 * sq(p[1] - sq(p[0])) + sq(p[0] - 1.0);
}

void jacmodbt7( double *p, double *jac, int m, int n, void *data )
{
	for( int i = 0; i < 5*m; i += 5 )
	{
		jac[i  ] = -400.0 * (p[1] - sq(p[0])) * p[0] + 2.0 * p[0] - 2.0;
		jac[i+1] =  200.0 * (p[1] - sq(p[0]));
		jac[i+2] =  0.0;
		jac[i+3] =  0.0;
		jac[i+4] =  0.0;
	}
}

/* Equilibrium combustion problem, constrained nonlinear equation from the book by Floudas et al.
 * Minimum at (0.0034, 31.3265, 0.0684, 0.8595, 0.0370)
 * constri: p[i]>=0.0001; (i=1..5)
 * constri+5: p[i]<=100.0; (i=1..5)
 */
void combust( double *p, double *x, int m, int n, void *data )
{
	constexpr double R   = 10;
	constexpr double R5  = 0.193;
	constexpr double R6  = 4.10622*1e-4;
	constexpr double R7  = 5.45177*1e-4;
	constexpr double R8  = 4.4975*1e-7;
	constexpr double R9  = 3.40735*1e-5;
	constexpr double R10 = 9.615*1e-7;

	x[0] = p[0] * p[1]      +
		   p[0] - 3.0 * p[4];
	x[1] = 2*p[0] * p[1]        +
		   p[0]                 +
		   3.0 * R10 * sq(p[1]) +
		   p[1] * sq(p[2])      +
		   R7 * p[1] * p[2]     +
		   R9 * p[1] * p[3]     +
		   R8 * p[1]            -
		   R * p[4];
	x[2] = 2 * p[1] * sq(p[2]) +
		   R7 * p[1] * p[2]    +
		   2 * R5 * sq(p[2])   +
		   R6 * p[2]           -
		   8 * p[4];
	x[3] = R9 * p[1] * p[3] +
		   2 * sq(p[3]) - 4 * R* p[4];
	x[4] = p[0] * p[1]      +
		   p[0]             +
		   R10 * sq(p[1])   +
		   p[1] * sq(p[2])  +
		   R7 * p[1] * p[2] +
		   R9 * p[1] * p[3] +
		   R8 * p[1]        +
		   R5 * sq(p[2])    +
		   R6 * p[2]        +
		   sq(p[3])         -
		   1.0;
}

void jaccombust(double *p, double *jac, int m, int n, void *data)
{
	constexpr double R   = 10;
	constexpr double R5  = 0.193;
	constexpr double R6  = 4.10622*1e-4;
	constexpr double R7  = 5.45177*1e-4;
	constexpr double R8  = 4.4975*1e-7;
	constexpr double R9  = 3.40735*1e-5;
	constexpr double R10 = 9.615*1e-7;

	std::fill( jac, jac + m*n, 0.0 );

	int j = 0;
	jac[  j] = p[1] + 1;
	jac[j+1] = p[0];
	jac[j+4] = -3;

	j += m;
	jac[  j] = 2 * p[1] + 1;
	jac[j+1] = 2 * p[0] + 6 * R10 * p[1] + sq(p[2]) + R7 * p[2] + R9 * p[3] + R8;
	jac[j+2] = 2 * p[1] * p[2] + R7 * p[1];
	jac[j+3] = R9 * p[1];
	jac[j+4] = -R;

	j += m;
	jac[j+1] = 2 * sq(p[2]) + R7 * p[2];
	jac[j+2] = 4 * p[1] * p[2] + R7 * p[1] + 4 * R5 * p[2] + R6;
	jac[j+4] = -8;

	j += m;
	jac[j+1] = R9 * p[3];
	jac[j+3] = R9 * p[1] + 4 * p[3];
	jac[j+4] = -4 * R;

	j += m;
	jac[j  ] = p[1] + 1;
	jac[j+1] = p[0] + 2 * R10 * p[1] + sq(p[2]) + R7 * p[2] + R9 * p[3] + R8;
	jac[j+2] = 2 * p[1] * p[2] + R7 * p[1] + 2 * R5 *p[2] + R6;
	jac[j+3] = R9 * p[1] + 2 * p[3];
}

/* Hock - Schittkowski (modified) problem 76 (linear inequalities & equations constrained), minimum at (0.0, 0.00909091, 0.372727, 0.354545)
 * The non-squared terms in the objective function have been removed, the rhs of constr2 has been changed to 0.4 (from 4)
 * and constr3 has been changed to an equation.
 *
 * constr1: p[0] + 2*p[1] + p[2] + p[3] <= 5;
 * constr2: 3*p[0] + p[1] + 2*p[2] - p[3] <= 0.4;
 * constr3: p[1] + 4*p[2] = 1.5;
 *
 */
void modhs76( double *p, double *x, int m, int n, void *data )
{
  x[0] = p[0];
  x[1] = std::sqrt(0.5) * p[1];
  x[2] = p[2];
  x[3] = std::sqrt(0.5) * p[3];
}

void jacmodhs76( double *p, double *jac, int m, int n, void *data )
{
  jac[ 0] = 1.0;
  jac[ 1] = 0.0;
  jac[ 2] = 0.0;
  jac[ 3] = 0.0;

  jac[ 4] = 0.0;
  jac[ 5] = std::sqrt(0.5);
  jac[ 6] = 0.0;
  jac[ 7] = 0.0;

  jac[ 8] = 0.0;
  jac[ 9] = 0.0;
  jac[10] = 1.0;
  jac[11] = 0.0;

  jac[12] = 0.0;
  jac[13] = 0.0;
  jac[14] = 0.0;
  jac[15] = std::sqrt(0.5);
}


#ifndef LMPP_HAVE_LAPACK
#ifdef _MSC_VER
#pragma message("LAPACK not available, some test problems cannot be used")
#else
#warning LAPACK not available, some test problems cannot be used
#endif // _MSC_VER
#endif /* LMPP_HAVE_LAPACK */


/* Rosenbrock function */
int prob__0( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { -1.2, 1.0 };
	x = {  0.0, 0.0 };

	int m = (int)p.size(); // 2
	int n = (int)x.size(); // 2

	int ret;
	if( alt_flag ) {
		/* with analytic Jacobian */
		ret = levmar_der( ros, jacros, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* no Jacobian */
		ret = levmar_dif( ros, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* modified Rosenbrock problem */
int prob__1( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { -1.2, 1.0 };
	x = {  0.0, 0.0, 0.0 };

	int m = (int)p.size(); // 2
	int n = (int)x.size(); // 3

	int ret;
	if( alt_flag ) {
		/* with analytic Jacobian */
		ret = levmar_der( modros, jacmodros, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* no Jacobian */
		ret = levmar_dif( modros, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* Powell's function */
int prob__2( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { 3.0, 1.0 };
	x = { 0.0, 0.0 };

	int m = (int)p.size(); // 2
	int n = (int)x.size(); // 2

	int ret;
	if( alt_flag ) {
		/* with analytic Jacobian */
		ret = levmar_der( powell, jacpowell, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* no Jacobian */
		ret = levmar_dif( powell, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr);
	}

	return ret;
}

/* Wood's function */
int prob__3( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { -3.0, -1.0, -3.0, -1.0 };
	x = {  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 };

	int m = (int)p.size(); // 4
	int n = (int)x.size(); // 6

	int ret;
	/* no Jacobian */
	ret = levmar_dif( wood, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	
	return ret;
}

/* Meyer's data fitting problem */
int prob__4( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { 8.85, 4.0, 2.5 };
	x = { 34.780, 28.610, 23.650, 19.630, 16.370, 13.720, 11.540, 9.744, 8.261, 7.030, 6.005, 5.147, 4.427, 3.820, 3.307, 2.872 };

	int m = (int)p.size(); // 3
	int n = (int)x.size(); // 16

	int ret;
	if( alt_flag ) {
		/* with analytic Jacobian */
		ret = levmar_der( meyer, jacmeyer, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* no Jacobian */
		std::vector<double> work(LMPP_DIF_WORKSZ(m,n));
		std::vector<double> covar(m*m);
		// no Jacobian, caller allocates work memory, covariance estimated
		ret = levmar_dif( meyer, p.data(), x.data(), m, n, 1000, opts, info, work.data(), covar.data(), nullptr );
		outs << "Covariance of the fit:\n";
		for( int i = 0; i < m; ++i )
		{
			for( int j = 0; j < m; ++j )
			{
				outs << covar[i*m+j] << " ";
			}
			outs << "\n";
		}
		outs << "\n";

	}

	/* uncomment the following block to verify Jacobian */
	double err[16];
	levmar_chkjac( meyer, jacmeyer, p.data(), m, n, nullptr, err );
	for( int i = 0; i < n; ++i )
	{
		outs << "gradient " << i << ", err " << err[i] << "\n", i, err[i];
	}
	outs << "\n";

	return ret;
}

/* Osborne's data fitting problem */
int prob__5( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { 0.5, 1.5, 1.0, 1.0E-2, 2.0E-2 };
	x = { 8.44E-1, 9.08E-1, 9.32E-1, 9.36E-1, 9.25E-1, 9.08E-1, 8.81E-1,
		  8.5E-1,  8.18E-1, 7.84E-1, 7.51E-1, 7.18E-1, 6.85E-1, 6.58E-1,
		  6.28E-1, 6.03E-1, 5.8E-1,  5.58E-1, 5.38E-1, 5.22E-1, 5.06E-1,
		  4.9E-1,  4.78E-1, 4.67E-1, 4.57E-1, 4.48E-1, 4.38E-1, 4.31E-1,
		  4.24E-1, 4.2E-1,  4.14E-1, 4.11E-1, 4.06E-1 };

	int m = (int)p.size(); // 5
	int n = (int)x.size(); // 33

	int ret;
	if( alt_flag ) {
		/* with analytic Jacobian */
		ret = levmar_der( osborne, jacosborne, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* no Jacobian */
		ret = levmar_dif( osborne, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* helical valley function */
int prob__6( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { -1.0, 0.0, 0.0 };
	x = {  0.0, 0.0, 0.0 };

	int m = (int)p.size(); // 3
	int n = (int)x.size(); // 3

	int ret;
	if( alt_flag ) {
		/* with analytic Jacobian */
		ret = levmar_der( helval, jachelval, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* no Jacobian */
		ret = levmar_dif( helval, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* Boggs-Tolle problem 3 */
int prob__7( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { 2.0, 2.0, 2.0, 2.0, 2.0 };
	x = { 0.0, 0.0, 0.0, 0.0, 0.0 };

	int m = (int)p.size(); // 5
	int n = (int)x.size(); // 5

	std::vector<double> A = { 1.0, 3.0, 0.0, 0.0,  0.0, 
							  0.0, 0.0, 1.0, 1.0, -2.0,
							  0.0, 1.0, 0.0, 0.0, -1.0 };
	std::vector<double> b = { 0.0, 0.0, 0.0 };

	int ret;
	if( alt_flag ) {
		/* lin. constraints, analytic Jacobian */
		ret = levmar_lec_der( bt3, jacbt3, p.data(), x.data(), m, n, A.data(), b.data(),
							  3, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* lin. constraints, no Jacobian */
		ret = levmar_lec_dif( bt3, p.data(), x.data(), m, n, A.data(), b.data(),
							  3, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* Hock - Schittkowski problem 28 */
int prob__8( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { -4.0, 1.0, 1.0 };
	x = {  0.0, 0.0, 0.0 };

	int m = (int)p.size(); // 3
	int n = (int)x.size(); // 3

	std::vector<double> A = { 1.0, 2.0, 3.0 };
	std::vector<double> b = { 1.0 };

	int ret;
	if( alt_flag ) {
		/* lin. constraints, analytic Jacobian */
		ret = levmar_lec_der( hs28, jachs28, p.data(), x.data(), m, n, A.data(), b.data(),
							  1, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* lin. constraints, no Jacobian */
		ret = levmar_lec_dif( hs28, p.data(), x.data(), m, n, A.data(), b.data(),
							  1, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* Hock - Schittkowski problem 48 */
int prob__9( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { 3.0, 5.0, -3.0, 2.0, -2.0 };
	x = { 0.0, 0.0,  0.0, 0.0,  0.0 };

	int m = (int)p.size(); // 5
	int n = (int)x.size(); // 5

	std::vector<double> A = { 1.0, 1.0, 1.0,  1.0,  1.0,
							  0.0, 0.0, 1.0, -2.0, -2.0 };
	std::vector<double> b = { 5.0, -3.0 };

	int ret;
	if( alt_flag ) {
		/* lin. constraints, analytic Jacobian */
		ret = levmar_lec_der( hs48, jachs48, p.data(), x.data(), m, n, A.data(), b.data(),
							  2, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* lin. constraints, no Jacobian */
		ret = levmar_lec_dif( hs48, p.data(), x.data(), m, n, A.data(), b.data(),
							  2, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* Hock - Schittkowski problem 51 */
int prob_10( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { 2.5, 0.5, 2.0, -1.0, 0.5 };
	x = { 0.0, 0.0, 0.0,  0.0, 0.0 };

	int m = (int)p.size(); // 5
	int n = (int)x.size(); // 5

	std::vector<double> A = { 1.0, 3.0, 0.0, 0.0,  0.0,
							  0.0, 0.0, 1.0, 1.0, -2.0,
							  0.0, 1.0, 0.0, 0.0, -1.0 };
	std::vector<double> b = { 4.0, 0.0, 0.0 };

	int ret;
	if( alt_flag ) {
		/* lin. constraints, analytic Jacobian */
		ret = levmar_lec_der( hs51, jachs51, p.data(), x.data(), m, n, A.data(), b.data(),
							  3, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* lin. constraints, no Jacobian */
		ret = levmar_lec_dif( hs51, p.data(), x.data(), m, n, A.data(), b.data(),
							  3, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* Hock - Schittkowski problem 01 */
int prob_11( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { -2.0, 1.0 };
	x = {  0.0, 0.0 };

	int m = (int)p.size(); // 2
	int n = (int)x.size(); // 2

	int ret;
	if( alt_flag ) {
		/* with analytic Jacobian */
		ret = levmar_der( hs01, jachs01, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		double lb[2], ub[2];

		lb[0] = -DBL_MAX;
		lb[1] = -1.5;
		ub[0] = ub[1] = DBL_MAX;

		/* with analytic Jacobian */
		ret = levmar_bc_der( hs01, jachs01, p.data(), x.data(), m, n, lb, ub, nullptr, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* Hock - Schittkowski (modified) problem 21 */
int prob_12( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { -1.0, -1.0 };
	x = {  0.0, 0.0 };

	int m = (int)p.size(); // 2
	int n = (int)x.size(); // 2

	int ret;
	if( alt_flag ) {
		/* with analytic Jacobian */
		ret = levmar_der( hs21, jachs21, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		double lb[2], ub[2];

		lb[0] =  2.0;
		lb[1] = -50.0;
		ub[0] =  50.0;
		ub[1] =  50.0;

		/* with analytic Jacobian */
		ret = levmar_bc_der( hs21, jachs21, p.data(), x.data(), m, n, lb, ub, nullptr, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* hatfldb problem */
int prob_13( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { 0.1, 0.1, 0.1, 0.1 };
	x = { 0.0, 0.0, 0.0, 0.0 };

	int m = (int)p.size(); // 4
	int n = (int)x.size(); // 4

	int ret;
	if( alt_flag ) {
		/* with analytic Jacobian */
		ret = levmar_der( hatfldb, jachatfldb, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		double lb[4], ub[4];

		lb[0] = lb[1] = lb[2] = lb[3] = 0.0;
		ub[0] = ub[2] = ub[3] = DBL_MAX;
		ub[1] = 0.8;

		/* with analytic Jacobian */
		ret = levmar_bc_der( hatfldb, jachatfldb, p.data(), x.data(), m, n, lb, ub, nullptr, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* hatfldc problem */
int prob_14( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { 0.9, 0.9, 0.9, 0.9 };
	x = { 0.0, 0.0, 0.0, 0.0 };

	int m = (int)p.size(); // 4
	int n = (int)x.size(); // 4

	int ret;
	if( alt_flag ) {
		/* with analytic Jacobian */
		ret = levmar_der( hatfldc, jachatfldc, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		double lb[4], ub[4];

		lb[0] = lb[1] = lb[2] = lb[3] =  0.0;
		ub[0] = ub[1] = ub[2] = ub[3] = 10.0;

		/* with analytic Jacobian */
		ret = levmar_bc_der( hatfldc, jachatfldc, p.data(), x.data(), m, n, lb, ub, nullptr, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* equilibrium combustion problem */
int prob_15( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { 0.0001, 0.0001, 0.0001, 0.0001, 0.0001 };
	x = { 0.0,    0.0,    0.0,    0.0,    0.0    };

	int m = (int)p.size(); // 5
	int n = (int)x.size(); // 5

	int ret;
	if( alt_flag ) {
		/* with analytic Jacobian */
		ret = levmar_der( combust, jaccombust, p.data(), x.data(), m, n, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		double lb[5], ub[5];

		lb[0] = lb[1] = lb[2] = lb[3] = lb[4] = 0.0001;
		ub[0] = ub[1] = ub[2] = ub[3] = ub[4] = 100.0;

		/* with analytic Jacobian */
		ret = levmar_bc_der( combust, jaccombust, p.data(), x.data(), m, n, lb, ub, nullptr, 5000, opts, info, nullptr, nullptr, nullptr );
	}
	
	return ret;
}

/* Hock - Schittkowski modified #1 problem 52 */
int prob_16( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { 2.0, 2.0, 2.0, 2.0, 2.0 };
	x = { 0.0, 0.0, 0.0, 0.0 };

	int m = (int)p.size(); // 5
	int n = (int)x.size(); // 4

	std::vector<double> A = { 1.0, 3.0, 0.0, 0.0,  0.0,
							  0.0, 0.0, 1.0, 1.0, -2.0,
							  0.0, 1.0, 0.0, 0.0, -1.0 };
	std::vector<double> b = { 0.0, 0.0, 0.0 };

	double lb[5], ub[5];

	lb[0] = -0.09;
	lb[1] =  0.0;
	lb[2] = -DBL_MAX;
	lb[3] = -0.2;
	lb[4] =  0.0;

	ub[0] =  DBL_MAX;
	ub[1] =  0.3;
	ub[2] =  0.25;
	ub[3] =  0.3;
	ub[4] =  0.3;

	/* penalty terms weights */
	double weights[5] = { 2000.0, 2000.0, 2000.0, 2000.0, 2000.0 };

	int ret;
	if( alt_flag ) {
		/* box & lin. constraints, analytic Jacobian */
		ret = levmar_blec_der( mod1hs52, jacmod1hs52, p.data(), x.data(), m, n, lb, ub, A.data(), b.data(),
							   3, weights, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* box & lin. constraints, no Jacobian */
		ret = levmar_blec_dif( mod1hs52, p.data(), x.data(), m, n, lb, ub, A.data(), b.data(),
							   3, weights, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* Schittkowski modified problem 235 */
int prob_17( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { -2.0, 3.0, 1.0 };
	x = {  0.0, 0.0 };

	int m = (int)p.size(); // 3
	int n = (int)x.size(); // 2

	std::vector<double> A = {  1.0, 0.0,  1.0,
							   0.0, 1.0, -4.0 };
	std::vector<double> b = { -1.0, 0.0 };

	double lb[3], ub[3];

	lb[0] = -DBL_MAX;
	lb[1] = 0.1;
	lb[2] = 0.7;

	ub[0] = DBL_MAX;
	ub[1] = 2.9;
	ub[2] = DBL_MAX;

	int ret;
	if( alt_flag ) {
		/* box & lin. constraints, analytic Jacobian */
		ret = levmar_blec_der( mods235, jacmods235, p.data(), x.data(), m, n, lb, ub, A.data(), b.data(),
								   2, nullptr, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* box & lin. constraints, no Jacobian */
		ret = levmar_blec_dif( mods235, p.data(), x.data(), m, n, lb, ub, A.data(), b.data(),
							   2, nullptr, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* Boggs & Tolle modified problem 7 */
int prob_18( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { -2.0, 1.0, 1.0, 1.0, 1.0 };
	x = {  0.0, 0.0, 0.0, 0.0, 0.0 };

	int m = (int)p.size(); // 5
	int n = (int)x.size(); // 5

	std::vector<double> A = { 1.0, 1.0, -1.0,  0.0, 0.0,
							  1.0, 1.0,  0.0, -1.0, 0.0,
							  1.0, 0.0,  0.0,  0.0, 1.0 };
	std::vector<double> b = { 1.0, 0.0, 0.5 };

	double lb[5], ub[5];

	lb[0] = -DBL_MAX;
	lb[1] = -DBL_MAX;
	lb[2] = -DBL_MAX;
	lb[3] = -DBL_MAX;
	lb[4] = -0.3;

	ub[0] =  0.7;
	ub[1] =  DBL_MAX;
	ub[2] =  DBL_MAX;
	ub[3] =  DBL_MAX;
	ub[4] =  DBL_MAX;

	int ret;
	if( alt_flag ) {
		/* box & lin. constraints, analytic Jacobian */
		ret = levmar_blec_der( modbt7, jacmodbt7, p.data(), x.data(), m, n, lb, ub, A.data(), b.data(),
							   3, nullptr, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* box & lin. constraints, no Jacobian */
		ret = levmar_blec_dif( modbt7, p.data(), x.data(), m, n, lb, ub, A.data(), b.data(),
							   3, nullptr, 10000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* Hock - Schittkowski modified #2 problem 52 */
int prob_19( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { 2.0, 2.0, 2.0, 2.0, 2.0 };
	x = { 0.0, 0.0, 0.0, 0.0, 0.0 };

	int m = (int)p.size(); // 5
	int n = (int)x.size(); // 5

	std::vector<double> C = { 1.0,  3.0, 0.0, 0.0,  0.0, 
							  0.0,  0.0, 1.0, 1.0, -2.0, 
							  0.0, -1.0, 0.0, 0.0,  1.0 };
	std::vector<double> d = { -1.0, -2.0, -7.0 };

	int ret;
	if( alt_flag ) {
		/* lin. ineq. constraints, analytic Jacobian */
		ret = levmar_bleic_der( mod2hs52, jacmod2hs52, p.data(), x.data(), m, n, nullptr, nullptr, nullptr, nullptr, 0,
								C.data(), d.data(), 3, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* lin. ineq. constraints, no Jacobian */
		ret = levmar_bleic_dif( mod2hs52, p.data(), x.data(), m, n, nullptr, nullptr, nullptr, nullptr, 0,
								C.data(), d.data(), 3, 1000, opts, info, nullptr, nullptr, nullptr );
	}

	return ret;
}

/* Hock - Schittkowski modified problem 76 */
int prob_20( std::ostream& outs, std::vector<double>& p, std::vector<double>& x, double* opts, double* info, bool alt_flag )
{
	p = { 0.5, 0.5, 0.5, 0.5 };
	x = { 0.0, 0.0, 0.0, 0.0 };

	int m = (int)p.size(); // 4
	int n = (int)x.size(); // 4

	std::vector<double> A = { 0.0, 1.0, 4.0, 0.0 };
	std::vector<double> b = { 1.5 };

	std::vector<double> C = { -1.0, -2.0, -1.0, -1.0,
							  -3.0, -1.0, -2.0,  1.0 };
	std::vector<double> d = { -5.0, -0.4 };

	double lb[4] = { 0.0, 0.0, 0.0, 0.0 };

	int ret;
	if( alt_flag ) {
		/* lin. ineq. constraints, analytic Jacobian */
		ret = levmar_bleic_der( modhs76, jacmodhs76, p.data(), x.data(), m, n, lb, nullptr, A.data(), b.data(),
								1, C.data(), d.data(), 2, 1000, opts, info, nullptr, nullptr, nullptr );
	} else {
		/* lin. ineq. constraints, no Jacobian */
		ret = levmar_bleic_dif( modhs76, p.data(), x.data(), m, n, lb, nullptr, A.data(), b.data(),
								1, C.data(), d.data(), 2, 1000, opts, info, nullptr, nullptr, nullptr );
	}
	/* variations:
	 * if no lb is used, the minimizer is (-0.1135922 0.1330097 0.3417476 0.07572816)
	 * if the rhs of constr2 is 4.0, the minimizer is (0.0, 0.166667, 0.333333, 0.0)
	 */

	return ret;
}

std::string map_reason(int reason)
{
	switch( reason )
	{
		case 1:
			return "small gradient J^T e";
		case 2:
			return "small Dp";
		case 3:
			return "itmax";
		case 4:
			return "singular matrix. Restart from current p with increased mu";
		case 5:
			return "no further error reduction is possible. Restart with increased mu";
		case 6:
			return "small ||e||_2";
		case 7:
			return "invalid (i.e. NaN or Inf) 'func' values. This is a user error";
		default:
			return "I don't know why";
	}
}

void output_results( std::ostream& outs, std::string const& probname, int ret, std::vector<double> const& info,
					 std::vector<double> const& p, std::vector<double> const& x )
{
	outs << "Results for " << probname.c_str() << ":\n";
	outs << "Levenberg-Marquardt returned " << ret << "\n\nSolution   p    = ";
	for( auto pi : p )
	{
		outs << pi << " ";
	}
	outs << "\nFunc value f(p) = ";
	for( auto xi : x )
	{
		outs << xi << " ";
	}
	outs << "\n\nMinimization info:\n";
	outs << "Initial ||e||_2       : " << info[0] <<"\n";
	outs << "Final   ||e||_2       : " << info[1] <<"\n";
	outs << "Final   ||J^T e||_inf : " << info[2] <<"\n";
	outs << "Final   ||Dp||_2      : " << info[3] <<"\n";
	outs << "Final   mu/max[J^T J] : " << info[4] <<"\n";
	outs << "# iterations          : " << info[5] <<"\n";
	outs << "reason " << info[6] << "              : " << map_reason(int(info[6])) << "\n";
	outs << "# function evaluations: " << info[7] <<"\n";
	outs << "# Jacobian evaluations: " << info[8] <<"\n";
	outs << "# lin systems solved  : " << info[9] <<"\n";
	outs << "\n";
}

void set_opts_info( std::vector<double>& opts, std::vector<double>& info )
{
	opts.resize(LMPP_OPTS_SZ);
	info.resize(LMPP_INFO_SZ);

	opts[0] = LMPP_INIT_MU;
	opts[1] = 1E-15; 
	opts[2] = 1E-15;
	opts[3] = 1E-20;
	// relevant only if the Jacobian is approximated using finite differences; specifies forward differencing 
	opts[4] = LMPP_DIFF_DELTA;
	// specifies central differencing to approximate Jacobian; more accurate but more expensive to compute!
	//opts[4] = -LMPP_DIFF_DELTA;

	for( int i = 5; i < LMPP_OPTS_SZ; ++i )
		opts[i] = 0.0;

	for( int i = 0; i < LMPP_INFO_SZ; ++i )
		info[i] = 0.0;
}

int main()
{
	std::vector<double> p;
	std::vector<double> x;

	std::vector<double> info;
	std::vector<double> opts;


	std::vector<const char*> probname = { "Rosenbrock function",
										  "modified Rosenbrock problem",
										  "Powell's function",
										  "Wood's function",
										  "Meyer's (reformulated) problem",
										  "Osborne's problem",
										  "helical valley function",
										  "Boggs & Tolle's problem #3",
										  "Hock - Schittkowski problem #28",
										  "Hock - Schittkowski problem #48",
										  "Hock - Schittkowski problem #51",
										  "Hock - Schittkowski problem #01",
										  "Hock - Schittkowski modified problem #21",
										  "hatfldb problem",
										  "hatfldc problem",
										  "equilibrium combustion problem",
										  "Hock - Schittkowski modified #1 problem #52",
										  "Schittkowski modified problem #235",
										  "Boggs & Tolle modified problem #7",
										  "Hock - Schittkowski modified #2 problem #52",
										  "Hock - Schittkowski modified problem #76" };

	std::vector<const char*> pfname = { "prob__0.log", "prob__1.log", "prob__2.log", "prob__3.log", "prob__4.log",
										"prob__5.log", "prob__6.log", "prob__7.log", "prob__8.log", "prob__9.log",
										"prob_10.log", "prob_11.log", "prob_12.log", "prob_13.log", "prob_14.log",
										"prob_15.log", "prob_16.log", "prob_17.log", "prob_18.log", "prob_19.log", "prob_20.log" };

	std::vector<int> problems = {  0, // Rosenbrock function
								   1, // modified Rosenbrock problem
								   2, // Powell's function
								   3, // Wood's function
								   4, // Meyer's (reformulated) problem
								   5, // Osborne's problem
								   6, // helical valley function
							#ifdef LMPP_HAVE_LAPACK
								   7, // Boggs & Tolle's problem 3
								   8, // Hock - Schittkowski problem 28
								   9, // Hock - Schittkowski problem 48
								  10, // Hock - Schittkowski problem 51
							#endif /* LMPP_HAVE_LAPACK */
								  11, // Hock - Schittkowski problem 01
								  12, // Hock - Schittkowski modified problem 21
								  13, // hatfldb problem
								  14, // hatfldc problem
								  15, // equilibrium combustion problem
							#ifdef LMPP_HAVE_LAPACK
		                          16, // Hock - Schittkowski modified #1 problem 52
		                          17, // Schittkowski modified problem 235
		                          18, // Boggs & Tolle modified problem #7
		                          19, // Hock - Schittkowski modified #2 problem 52
		                          20, // Hock - Schittkowski modified problem #76"
							#endif /* LMPP_HAVE_LAPACK */
								 };

	std::vector<tfunc_ptr> tfuncs = { prob__0, prob__1, prob__2, prob__3, prob__4, prob__5, prob__6,
									  prob__7, prob__8, prob__9, prob_10, prob_11, prob_12, prob_13,
									  prob_14, prob_15, prob_16, prob_17, prob_18, prob_19, prob_20 };

	for( std::size_t i = 0; i < problems.size(); ++i )
	{
		int problem_id    = problems[i];
		const char* fname = pfname  [problem_id];
		std::string pname = probname[problem_id];
		tfunc_ptr tf      = tfuncs  [problem_id];

		std::ofstream outf(fname);

		set_opts_info( opts, info );
		int ret = (*tf)( outf, p, x, opts.data(), info.data(), true );
		output_results( outf, pname, ret, info, p, x );

		outf.close();
	}

  return 0;
}
