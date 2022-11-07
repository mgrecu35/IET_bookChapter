/* multiscatter_ascii.c -- Single-profile interface for lidar multiple
   scattering algorithm

   Copyright (C) 2004-2011 Robin Hogan <r.j.hogan@reading.ac.uk> 

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


   Run "./multiscatter -help" to get usage information.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#define _GNU_SOURCE 1
#include <fenv.h>

#include "multiscatter.h"

#define MAX_CHARS 128

#ifdef SINGLE_PRECISION
#define READ_HEADER_FORMAT "%d %g %g %g %g\n"
#define READ_FORMAT "%g %g %g %g %g %g %g %g %g %g %n\n"
#else
#define READ_HEADER_FORMAT "%d %lg %lg %lg %lg\n"
#define READ_FORMAT "%lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %n\n"
#endif

#define CHECK(function) if ((status = (function))) { fprintf(stderr, \
   "Error at line %d of %s: code %d\n", __LINE__, __FILE__, status); exit(status); }


/* Print usage information */
static
void
usage(char *exec_name)
{
  fprintf(stderr,
	  "Usage\n"
	  "  %s [options] data_in.dat > data_out.dat\n"
	  "  %s [options] < data_in.dat > data_out.dat\n"
	  "\n", exec_name, exec_name);
  fprintf(stderr, 
	  "General options\n" 
	  "  -help           Display this message\n"
	  "  -repeat n       Repeat algorithm n times (for benchmarking)\n"
	  "  -quiet          Don't report activity to stderr\n"
	  "  -v1             Use version 1.x interpretation of first line of input file\n"
	  "  -auto           Automatically select algorithm settings from wavelength etc\n"
	  "  -radar          Use settings appropriate for radar\n"
	  "  -lidar          Use settings appropriate for lidar\n"
	  "  -algorithms <small_angle_algorithm> <wide_angle_algorithm>\n"
	  "                  Manually select algorithm, where <small_angle_algorithm> is:\n"
	  "      none     - No single or small-angle scattering\n"
	  "      single   - Single scattering (no small-angle scattering)\n"
	  "      original - Original Hogan (2006) algorithm: speed O(N^2)\n"
	  "      fast     - Faster Hogan (2008) algorithm: speed O(N) DEFAULT\n"
	  "      explicit - Eloranta-like explicit representation of each order of scattering\n"
	  "      lag      - Hogan (2008) but also with lag calculation (m)\n"
	  "                  ...and where <wide_angle_algorithm> is:\n"
	  "      none     - No wide-angle scattering\n"
	  "      tdts     - Time-dependent two stream (Hogan and Battaglia 2008)\n"
	  "      lidar    - TDTS assuming forward lobe DEFAULT\n"
	  "      radar    - TDTS assuming no forward lobe\n"
	  /* 
	  "  -single-only    Single scattering only\n"
	  "  -sa-only        Don't include wide-angle multiple scattering\n"
	  "  -qsa-only       (alternative to \"-sa-only\")\n"
	  "  -wide-only      Only wide-angle multiple scattering\n"
	  */
	  "  -hsrl           Output particulate and air backscatter separately\n"
	  "  -gaussian-receiver\n"
	  "                  Receiver is Gaussian rather than top-hat shaped\n"
	  "  -jacobian       Output the approximate but fast Jacobian\n"
	  "  -numerical-jacobian\n"
	  "                  Output the Jacobian calculated (slowly) using finite\n"
	  "                  differences\n"
	  "  -ext-only       Only calculate the Jacobian with respect to extinction\n"
	  "  -adjoint        Output the adjoint as well\n"
	  "  -check-adjoint  Calculate the adjoint and check it with a numerical Jacobian\n"
	  "  -annular-detectors\n"
	  "                  2nd and subsequent detectors are ring shaped\n"
	  "\n");
  fprintf(stderr,
	  "Options for ORIGINAL small-angle (SA) algorithm\n"
	  /*
	  "  -fast-sa        Use fast O(N) SA model\n"
	  "  -fast-qsa       (alternative to \"-fast-sa\")\n"
	  "  -lag            Calculate SA lag (m) - EXPERIMENTAL\n"
	  */
	  "  -simple-optical-depth\n"
	  "                  Use simple optical depth integration\n"
	  "  -crude-optical-depth\n"
	  "                  Use crude optical depth integration\n"
	  "  -crude-integration\n"
	  "                  Use crude double/multiple scattering integration\n");
  fprintf(stderr,
	  "  -no-multiscat-within-gate\n"
	  "                  Photon cannot be forward scattered more than once within a\n"
	  "                  single gate; note that this has the opposite effect to\n"
	  "                  \"-crude-integration\"\n"
	  "  -double-scattering-only\n"
	  "                  Don't include triple and higher-order scatterings\n");
  fprintf(stderr,
	  "  -no-molecular-extinction\n"
	  "                  Molecules do not extinguish the beam\n"
	  "  -wide-angle-cutoff <theta>\n"
	  "                  Forward scattering at angles greater than <theta> radians\n"
	  "                  are deemed to escape, a crude way to deal with a problem\n"
	  "                  associated with aerosols\n");
  fprintf(stderr,
	  "  -crude-double-scattering\n"
	  "                  Don't use Eloranta's slow but accurate double scattering\n"
	  "                 formulation\n"
	  "  -approx-exp     Appriximate the exp() function call for speed\n"
	  "\n");
  fprintf(stderr,
	  "Options for EXPLICIT small-angle (SA) algorithm\n"
	  "  -explicit n     Use an explicit model with n orders of scattering\n"
	  "  -output-distribution n dx\n"
	  "                  Output horizontal photon distributions at n points spaced\n"
	  "                  dx apart, starting at dx/2\n"
	  "\n");
  fprintf(stderr,
	  "Options for wide-angle multiple-scattering\n"
	  /*
	  "  -no-forward-lobe\n"
	  "                  Radar-like phase function behaviour: use single-scattering\n"
	  "                  rather than SA\n"
	  */
	  "  -ignore-source-widening\n"
	  "                  Ignore widening effect of small-angle scattering on source beam for wide-angle calc\n"
	  "  -optimize-wide-angle-gates\n"
	  "                  Skip optically thin gates in wide-angle calculation\n"
	  "  -simple-2s-coeffts\n"
	  "                  Use the simple upwind Euler formulation (can be unstable\n"
	  "                  for high optical depth)\n");
  fprintf(stderr,
	  "  -ssa-scales-forward-lobe\n"
	  "                  Single-scattering albedo less than unity reduces the\n"
	  "                  forward scattering lobe as well as wide-angle scattering\n"
	  "  -num-samples m  Output m samples, allowing sampling of signals appearing\n"
	  "                  to originate below ground\n"
	  "  -propagation-to-stderr\n"
	  "                  Print the diffuse photon energies and variances to\n"
	  "                  standard error\n"
	  "\n");
  fprintf(stderr,
	  "Input data\n"
	  "  First line: 5 values\n"
	  "    1: Number of range gates in profile\n"
	  "    2: Wavelength (m)\n"
	  "    3: Altitude of instrument (m)\n"
	  "    4: Transmitter divergence, 1/e half-width (radians)\n"
	  "    5: Receiver field-of-view, half-width of first receiver (radians)\n"
	  "   6+: Fields-of-view for any other channels (radians)\n");
  fprintf(stderr,
	  "  (note that with the \"-v1\" option, the version 1.x file format is assumed,\n"
	  "   in which the order is (1) number of range gates, (2) wavelength, (3) divergence,\n"
	  "   (4) field-of-view, (5) altitude, and only one field of view is allowed)\n");
  fprintf(stderr,
	  "  Subsequent lines: 4, 5, 8, 10 or more values (all 8 required for the wide-angle\n"
	  "  calculation; all 10 to represent anisotropic phase functions near 180 deg);\n"
	  "  more to calculate the adjoint\n"
	  "    1: Range of gate above ground starting with nearest gate to instrument (m)\n"
	  "    2: Extinction coefficient of cloud/aerosol only (m-1)\n"
	  "    3: Equivalent-area cloud/aerosol radius (m)\n"
	  "    4: Extinction-to-backscatter ratio of cloud/aerosol (sr)\n"
	  "    5: Air extinction coefficient (m-1); optional (zero if missing)\n");
  fprintf(stderr,
	  "    6: Single scattering albedo of cloud/aerosol\n"
	  "    7: Scattering asymmetry factor of cloud/aerosol\n"
	  "    8: Single scattering albedo of air\n");
  fprintf(stderr,
	  "    9: Fraction of cloud/aerosol backscatter due to droplets (real number\n"
	  "         between 0 and 1), for representing anisotropic near-180 phase\n"
	  "         functions in the SA algorithm\n"
	  "   10: Pristine ice fraction, for anisotropic near-180 phase functions\n"
	  "         from Yang et al. (2000); note that irregular ice particles\n"
	  "         are believed to have isotropic near-180 phase functions, in\n"
	  "         which case set this variable to zero\n");
  fprintf(stderr, 
	  "  Subsequent lines (only with \"-adjoint\" option)\n"
	  " 11 to 10+n: Adjoint input for particulate backscatter of the n fields-of-view\n"
	  " 11+n: Adjoint input for air backscatter of the first field-of-view (for HSRL)\n"
	  "\n");
  fprintf(stderr,
	  "Output data (without -jacobian or -numerical-jacobian option)\n"
	  "  Default: 5 values per range gate\n"
	  "    1: Index of range gate\n"
	  "    2: Apparent range above ground (m)\n"
	  "    3: Extinction coefficient (m-1)\n"
	  "    4: Equivalent-area particle radius (microns)\n"
	  "    5: Apparent backscatter (m-1 sr-1), for particulates only if with \"-hsrl\"\n");
  fprintf(stderr,
	  "  With the \"-hsrl\" option\n"
	  "    6: Apparent backscatter of the air only (m-1 sr-1)\n");
  fprintf(stderr,
	  "  With the \"-adjoint\" option (where n=6 if \"-hsrl\", n=5 otherwise)\n"
	  "  n+1: Adjoint of extinction coefficient (m)\n"
	  "  n+2: Adjoint of single-scattering albedo\n"
	  "  n+3: Adjoint of asymmetry factor\n"
	  "  n+4: Adjoint of extinction-to-backscatter ratio (m sr)\n"
	  "\n");
  fprintf(stderr,
	  "Output data (with -jacobian or -numerical-jacobian option)\n"
	  "   n x m matrix: d(attenuated backscatter) / d(extinction coefficient)\n"
	  "and if \"-ext-only\" is not set:\n"
	  "   n x m matrix: d(attenuated backscatter) / d(single-scattering albedo)\n"
	  "   n x m matrix: d(attenuated backscatter) / d(asymmetry factor)\n"
	  "   n x m matrix: d(attenuated backscatter) / d(particle radius)\n"
	  "   1 x m vector: d(attenuated backscatter) / d(ext-to-bscat ratio)\n");
}

static
void
print_error_AD(FILE* file, ms_real AD, ms_real AD_test) {
  //#define PANTS 1
#ifndef PANTS
  if (AD == 0.0) {
    if (AD_test == 0.0) {
      fprintf(file, "   0      ");
    }
    else {
      fprintf(file, "%+g\t", AD_test);
    }
  }
  else {
    fprintf(file, "%+8.3f%% ", 100.0*(AD_test-AD)/AD);
  }
#else
  fprintf(file, "(%g,%g)\t", AD, AD_test);

#endif
}


/* Main program */

/*multiscatter(nrange, extinct, ext2bscatt, salb, g, &
     bscatt, lambd,noMS,theta,dr)*/
int multiscatter_(int *nrange, float *extFort, 
		  float *ext2bscatt, float *salbFort, float *gFort,
		  float *bscatFort, float *lambd, int *noMS, float *theta, float *dr)
{
  /* VARIABLE DECLARATIONS */

  FILE *infile;// = fopen("msRProf2.in","r");
  int n, m = 0, i, iarg;
  int nrepeats = 1;
  int output_jacobian = 0;
  int output_adjoint = 0;
  int calc_jacobian = 0;
  int separate_bscat_air = 0;
  int use_isotropic_pp = 0;
  int use_air_ext = 1;
  int automatically_configure = 0;
  int automatically_configure_tdts = 0;
  int manually_select_algorithms = 0;
  int manually_select_cbh = 0;
  int print_stats = 0;
  int use_version_1x = 0;
  int ninputs;
  int norder = 4;
  char buffer[MAX_CHARS];
  int ch, status;
  char* strbuf = NULL;
  char* curbuf = NULL;
  char* newcurbuf = NULL;
  size_t strbufoffset = 0;
  size_t strbuflen = 0;
  ms_real rho_receiver_old = 1.0;

  ms_real* range = NULL;
  ms_real* radius = NULL;
  ms_real* ext = NULL;
  ms_real* ext_bscat_ratio = NULL;
  ms_real* ext_air = NULL;
  ms_real* droplet_fraction = NULL;
  ms_real* pristine_ice_fraction = NULL;
  ms_real* ssa = NULL;
  ms_real* g = NULL;
  ms_real* ssa_air = NULL;

  ms_real* bscat = NULL;
  ms_real* bscat_air = NULL;

  ms_real* ext_AD = NULL;
  ms_real* ssa_AD = NULL;
  ms_real* g_AD = NULL;
  ms_real* radius_AD = NULL;
  ms_real* ext_bscat_ratio_AD = NULL;
 
  ms_real* bscat_AD = NULL;
  ms_real* bscat_air_AD = NULL;

  ms_config config = MS_DEFAULT_CONFIG;
  ms_instrument instrument = MS_DEFAULT_INSTRUMENT;
  ms_surface surface = MS_DEFAULT_SURFACE;

  /* Enable some exceptions. At startup all exceptions are masked. */
   //   feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
   //FE_DIVBYZERO, FE_INEXACT, FE_INVALID, FE_OVERFLOW, FE_UNDERFLOW 
   //  feenableexcept(FE_ALL_EXCEPT);
  //  fesetround(FE_UPWARD);
  /* HANDLE COMMAND-LINE ARGUMENTS */

  config.small_angle_algorithm = MS_SINGLE_SCATTERING;
  config.wide_angle_algorithm = MS_WIDE_ANGLE_NONE;
  automatically_configure_tdts = 1;
  //config.small_angle_algorithm = MS_SINGLE_SCATTERING;
  //config.wide_angle_algorithm = MS_WIDE_ANGLE_TDTS_NO_FORWARD_LOBE;
  separate_bscat_air = 1;

  /* REPORT WHAT IS BEING DONE */

  if (!(config.options & MS_QUIET)) {
    //fprintf(stderr, "Multiscatter %s\n", MS_VERSION);
    if (infile == stdin) {
      fprintf(stderr, "Reading stdin...\n");
    }
    else {
      //fprintf(stderr, "Reading ...");
    }
  }

  /* READ AND CHECK THE INPUT FILE */

  /* Skip commented lines */
  /*
    while ((ch = fgetc(infile)) == '#') {
    while ((ch = fgetc(infile)) != '\n') {
    if (ch == EOF) {
    fprintf(stderr, "Error: only comments found in input file\n");
    exit(MS_INPUT_FILE_ERROR);
    }
    }
    }*/
  // ungetc(ch, infile);

  /* Read input data */
  /* First line can be any length, in principle: read it into strbuf */

  
  instrument.nfov=1;
  n=80;
  n=*nrange;
  instrument.rho_receiver=malloc(sizeof(ms_real)*(instrument.nfov));
  instrument.wavelength=0.00857;
  instrument.altitude=400000;
  instrument.rho_receiver[0]=0.0062173;
  //printf("%g \n",instrument.rho_receiver[0]);
  instrument.rho_receiver[0]=0.0062173;
  instrument.rho_transmitter=0.0062173;

  /* Allocate memory */
  range = malloc(sizeof(ms_real)*n);
  radius = malloc(sizeof(ms_real)*n);
  ext = malloc(sizeof(ms_real)*n);
  ext_bscat_ratio = malloc(sizeof(ms_real)*n);
  ext_air = malloc(sizeof(ms_real)*n);
  ssa = malloc(sizeof(ms_real)*n);
  ssa_air = malloc(sizeof(ms_real)*n);
  droplet_fraction = malloc(sizeof(ms_real)*n);
  pristine_ice_fraction = malloc(sizeof(ms_real)*n);
  g = malloc(sizeof(ms_real)*n);

  if (!range || !radius || !ext || !ext_bscat_ratio || !ext_air 
      || !ssa || !ssa_air || !droplet_fraction || !pristine_ice_fraction || !g) {
    fprintf(stderr, "Error allocating space for input variables\n");
    exit(MS_MEMORY_ALLOCATION_ERROR);
  }


  /* Read in the n data levels */
  for (i = 0; i < n; i++) {
    /* float *ext2bscatt, float *salbFort, float *gFort,
       float *bscatFort*/
    radius[i]=0.001;
    range[i]=(n-1-i)*(*dr)+(*dr)/2;
    ext[i]=extFort[i];
    ext_bscat_ratio[i]=ext2bscatt[i];
    ext_air[i]=0;
    ssa[i]=salbFort[i];
    g[i]=gFort[i];
    ssa_air[i]=0;
    droplet_fraction[i]=0;
    pristine_ice_fraction[i]=0;
      /*int nchars = 0;
	fgets(buffer, MAX_CHARS, infile);
	ninputs = sscanf(buffer, READ_FORMAT, range+i, ext+i,
	radius+i, ext_bscat_ratio+i, ext_air+i,
	ssa+i, g+i, ssa_air+i,
	droplet_fraction+i, pristine_ice_fraction+i,
	&nchars);
	//printf("%g \n",range[i]);
    */
  }
  for (i = 0; i < -n; i++)
    printf("%g \n",range[i]);

  if (automatically_configure) {
    if (instrument.wavelength > MS_RADAR_LIDAR_TRANSITION_WAVELENGTH) {
      /* Assume we have a radar */
      if (!manually_select_algorithms) {
	config.small_angle_algorithm = MS_SINGLE_SCATTERING;
      }
      if ((!manually_select_algorithms) || automatically_configure_tdts) {
	config.wide_angle_algorithm = MS_WIDE_ANGLE_TDTS_NO_FORWARD_LOBE;
      }
      if (!manually_select_cbh) {
	config.coherent_backscatter_enhancement = 2.0;
      }
      instrument.receiver_type = MS_GAUSSIAN;
    }
    else {
      /* Assume we have a lidar */
    }
  }
  else if (automatically_configure_tdts) {
    if (instrument.wavelength > MS_RADAR_LIDAR_TRANSITION_WAVELENGTH) {
      config.wide_angle_algorithm = MS_WIDE_ANGLE_TDTS_NO_FORWARD_LOBE;
    }
  }
  
  if(*noMS==1)
    config.wide_angle_algorithm = MS_WIDE_ANGLE_NONE;
  /* Allocate space for output */
  m=n;
  bscat = malloc(sizeof(ms_real)*instrument.nfov*m);
  if (separate_bscat_air) {
    bscat_air = malloc(sizeof(ms_real)*instrument.nfov*m);
  }
  if (!bscat || ((!bscat_air) && separate_bscat_air) ) {
    fprintf(stderr, "Error allocating space for apparent backscatter output\n");
    exit(MS_MEMORY_ALLOCATION_ERROR);
  }


  /* REPORT WHAT CALCULATIONS ARE BEING PERFORMED */
  FILE *fout;
  /*  fout=fopen("junk2","w");
  ms_print_algorithms(config, instrument, range, use_isotropic_pp,
		      separate_bscat_air, output_adjoint, output_jacobian,
		      calc_jacobian, fout);
  
		      fclose(fout);*/
  /**/
  if (m < n) {
    //    fprintf(stderr, "Warning: currently the number of samples (%d) cannot be fewer than\n"
    //	    "   the number of data points; setting them to be equal\n");*/
    n = m;
  }

  if (instrument.receiver_type == MS_GAUSSIAN
      && (config.small_angle_algorithm == MS_SMALL_ANGLE_PVC_ORIGINAL
	  || config.small_angle_algorithm == MS_SMALL_ANGLE_PVC_EXPLICIT)) {
    fprintf(stderr, "Warning: PVC algorithm will use a top-hat receiver pattern\n");
  }
  if (instrument.wavelength > MS_RADAR_LIDAR_TRANSITION_WAVELENGTH 
      && config.small_angle_algorithm > MS_SINGLE_SCATTERING) {
    fprintf(stderr, "Warning: wavelength greater than %g microns yet using small-angle scattering;\n"
	    "  should the \"-auto\" option be specified?\n",
	    MS_RADAR_LIDAR_TRANSITION_WAVELENGTH*1.0e6);
  }
  if (instrument.wavelength > MS_RADAR_LIDAR_TRANSITION_WAVELENGTH 
      && config.wide_angle_algorithm == MS_WIDE_ANGLE_TDTS_FORWARD_LOBE) {
    fprintf(stderr, "Warning: wavelength greater than %g microns yet using wide-angle scattering\n"
	    "  assuming a forward lobe; should the \"-auto\" option be specified?\n",
	    	    MS_RADAR_LIDAR_TRANSITION_WAVELENGTH*1.0e6);
  }
  //  printf("%g \n",MS_RANGE_SPACING_TOLERANCE);
 

  /* CHECK RANGE-GATE SPACING */
  if ((!ms_range_spacing_is_regular(n, range, 
				   MS_RANGE_SPACING_TOLERANCE))
      && config.wide_angle_algorithm != MS_WIDE_ANGLE_NONE) {
    fprintf(stderr, 
	    "  The range-gate spacing is not constant to within a tolerance of 5%%,\n"
	    "  so will be interpolated on to a regular grid for the purposes of wide-angle scattering\n");
  }

  /* RUN ALGORITHM */
//  if (config.options & MS_CRUDE_INTEGRATION) {
//    ext_air = ssa_air = NULL;
//  }
  CHECK(multiscatter(n, m, &config, instrument, surface, range, radius,
			   ext, ssa, g, ext_bscat_ratio, ext_air, ssa_air,
			   droplet_fraction, pristine_ice_fraction,
			   bscat, bscat_air));
  config.options |= MS_QUIET;	  
  

   
  for (i = 0; i < n; i++)
    bscatFort[i]=bscat[i];
  for (i = 0; i < -n; i++) {
    int ifov;
    /* Print the basic variables */
    fprintf(stdout, "%d %g %g %g %14.9g\n",
	    i+1, 
	    range[i],
	    ext[i],
	    radius[i],
	    bscat[i]);

  }

  if (separate_bscat_air) 
    free(bscat_air);
 
  free(range);
  free(radius);
  free(ext);
  free(ext_bscat_ratio);
  free(ext_air);
  free(ssa);
  free(ssa_air);
  free(droplet_fraction);
  free(pristine_ice_fraction);
  free(g);
}
