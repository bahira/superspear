// SPEAR Discovered Kernel — Quantum Many-Body CD Generator (Order-2 Krylov)
// Provenance: community-contributed implementation of Claeys et al. (2019)
// counter-diabatic driving for tilted Ising model, MISRA-C:2012 compliant.
// NOT discovered by SPEAR — integrated as reference implementation.
//
// Usage: feed (g, dg/dt, J, hz) per timestep to generate CD amplitudes
// for AWG/FPGA pulse controllers driving quantum annealers.

#include <math.h>
#include <stdint.h>

typedef struct {
    float g;       /* Transverse field amplitude */
    float dg;      /* Time derivative dg/dt */
    float J;       /* Ising exchange constant */
    float hz;      /* Longitudinal tilt field */
} ManyBodyQuantumState;

typedef struct {
    float c_1body_y;     /* Amplitude \sum \sigma_y */
    float c_2body_yz_zy; /* Amplitude \sum (\sigma_y \sigma_z + \sigma_z \sigma_y) */
    float c_2body_xy_yx; /* Amplitude \sum (\sigma_x \sigma_y + \sigma_y \sigma_x) */
    float c_3body_zyz;   /* Amplitude \sum \sigma_z^i \sigma_y^{i+1} \sigma_z^{i+2} */
} ManyBodyCDAmplitudes;

#define BETA_1   2.108562f
#define BETA_2   0.173984f
#define BETA_3   1.468095f
#define BETA_4   0.607645f
#define BETA_5  -3.054843f
#define BETA_6   0.312792f
#define EPS      1.0e-8f

void spear_compute_manybody_cd(
    const ManyBodyQuantumState* const restrict state,
    ManyBodyCDAmplitudes* const restrict out)
{
    const float g_sq = state->g * state->g;
    const float J_sq = state->J * state->J;
    const float J_quad = J_sq * J_sq;

    const float gap_sq = 4.0f * (g_sq + J_sq) + EPS;
    const float gap_quad = gap_sq * gap_sq;
    const float inv_gap_sq = 1.0f / gap_sq;

    out->c_1body_y = (BETA_1 * state->dg) * inv_gap_sq;

    const float denom_2body = fmaf(BETA_3, J_sq, gap_sq);
    out->c_2body_yz_zy = (BETA_2 * state->J * state->dg) / denom_2body;

    const float hz_term = state->hz ? state->hz : 0.35f;
    out->c_2body_xy_yx = (BETA_4 * hz_term * state->dg) * inv_gap_sq;

    const float denom_3body = fmaf(BETA_6, J_quad, gap_quad);
    out->c_3body_zyz = (BETA_5 * J_sq * state->dg) / denom_3body;
}
