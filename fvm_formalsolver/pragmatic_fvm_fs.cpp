#include "Lightweaver.hpp"
#include "LwInternal.hpp"
#include <cstdio>

enum class AloMode {
    FullCellTransfer,
    AverageCellTransfer,
    ImidOnly
};

void pragmatic_1_fvm_1d_impl(
    LwInternal::FormalData* fd,
    f64 z_mu,
    bool to_obs,
    f64 I_start
) {
    JasUnpack((*fd), chi, S, Psi, I, atmos);
    JasUnpack((*atmos), height, Nspace);
    const bool compute_operator = bool(Psi);
    constexpr f64 TrapezThreshold = 1.0;
    // NOTE(cmo): This is the only correct solution for the Lambda* operator to
    // show the action of Sk on J
    constexpr AloMode alo_mode = AloMode::ImidOnly;

    int dk = -1;
    int k_start = Nspace - 1;
    int k_end = 0;
    if (!to_obs) {
        dk = 1;
        k_start = 0;
        k_end = Nspace - 1;
    }

    auto integrate_cell = [&](f64 I_upw, f64 z_km, f64 z_kp, int k) {
        const f64 chi_k = chi(k);
        const f64 Sk = S(k);
        const f64 dtau_km = z_km * chi_k;
        const f64 dtau_kp = z_kp * chi_k;
        const f64 dtau = dtau_km + dtau_kp;

        f64 Is = I_upw;
        f64 I_mid = Is;
        f64 Psi_k = 0.0;
        bool sc = true;
        if (dtau < TrapezThreshold) {
            sc = false;
            I_mid = (I_upw + dtau_km * Sk) / (1.0 + dtau_km);
            I(k) = I_mid;
            I_upw = dtau_kp * Sk + (1.0 - dtau_kp) * I_mid;
            if (compute_operator) {
                if constexpr (alo_mode == AloMode::FullCellTransfer) {
                    // Full transfer through cell
                    Psi_k = ((1.0 - dtau_kp) * dtau_km / (1.0 + dtau_km) + dtau_kp) / chi_k;
                } else if constexpr (alo_mode == AloMode::AverageCellTransfer) {
                    // Average of two halves of cell
                    Psi_k = 0.5 * (dtau_km / (1.0 + dtau_km) + dtau_kp) / chi_k;
                } else if constexpr (alo_mode == AloMode::ImidOnly) {
                    // Contribution to I_mid
                    Psi_k = dtau_km / (1.0 + dtau_km) / chi_k;
                }
            }
        } else {
            const f64 edt_km = std::exp(-dtau_km);
            const f64 one_m_edt_km = -std::expm1(-dtau_km);
            const f64 edt_kp = std::exp(-dtau_kp);
            const f64 one_m_edt_kp = -std::expm1(-dtau_kp);
            f64 I_mid = I_upw * edt_km + Sk * one_m_edt_km;
            I(k) = I_mid;
            I_upw = I_mid * edt_kp + Sk * one_m_edt_kp;
            if (compute_operator) {
                if constexpr (alo_mode == AloMode::FullCellTransfer) {
                    // Full transfer through cell
                    Psi_k = (one_m_edt_km * edt_kp + one_m_edt_kp) / chi_k;
                    // Psi_k = (1.0 - edt_km * edt_kp) / chi_k;
                } else if constexpr (alo_mode == AloMode::AverageCellTransfer) {
                    // Average of two halves of cell
                    Psi_k = 0.5 * (one_m_edt_km + one_m_edt_kp) / chi_k;
                } else if constexpr (alo_mode == AloMode::ImidOnly) {
                    // Contribution to I_mid
                    Psi_k = one_m_edt_km / chi_k;
                }
            }
        }
        Psi(k) = Psi_k;
        return I_upw;
    };

    f64 I_upw = I_start;
    // NOTE(cmo): We are at I(k_start - 1/2*dk)
    // z(k-1/2dk)
    // NOTE(cmo): z_km and z_kp are actually Delta z between the cell interface
    // and centre.
    int k = k_start;
    f64 z_km = 0.5 * std::abs(height(k) - height(k + dk)) * z_mu;
    f64 z_kp = std::abs(0.5 * (height(k) + height(k + dk)) - height(k)) * z_mu;
    I_upw = integrate_cell(I_upw, z_km, z_kp, k);


    // NOTE(cmo): Inner cells
    for (k = k_start + dk; k != k_end; k += dk) {
        z_km = std::abs(0.5 * (height(k) + height(k - dk)) - height(k)) * z_mu;
        z_kp = std::abs(0.5 * (height(k) + height(k + dk)) - height(k)) * z_mu;

        I_upw = integrate_cell(I_upw, z_km, z_kp, k);
    }

    // NOTE(cmo): Final cell
    k = k_end;
    z_km = std::abs(0.5 * (height(k) + height(k - dk)) - height(k)) * z_mu;
    z_kp = 0.5 * std::abs(height(k) - height(k - dk)) * z_mu;
    I_upw = integrate_cell(I_upw, z_km, z_kp, k);
    // NOTE(cmo): There may be a half-cell error for the outgoing radiation if the
    // top cell is optically thick, but that seems like an unlikely scenario.
}

void pragmatic_1_fvm_1d(
    LwInternal::FormalData* fd,
    int la,
    int mu,
    bool to_obs,
    const F64View1D& wave
) {
    JasUnpack((*fd), atmos, chi);
    const f64 wav = wave(la);

    const f64 z_mu = 1.0 / atmos->muz(mu);
    const auto& height = atmos->height;

    int dk = -1;
    int k_start = atmos->Nspace - 1;
    auto bc = atmos->zLowerBc;
    if (!to_obs) {
        dk = 1;
        k_start = 0;
        bc = atmos->zUpperBc;
    }
    f64 start_temperature[2] = {atmos->temperature(k_start), atmos->temperature(k_start + dk)};

    const f64 dtau_uw = (
        0.5 * z_mu * (chi(k_start) + chi(k_start + dk))
        * std::abs(height(k_start) - height(k_start + dk))
    );

    f64 Iupw = 0.0;
    if (bc.type == THERMALISED) {
        f64 Bnu[2];
        planck_nu(2, start_temperature, wav, Bnu);
        // NOTE(cmo): This isn't technically quite right for the FVM setup, but
        // it's close enough, and we're not actually planning on really using
        // the diffusion approximation boundary here.
        Iupw = Bnu[0] - (Bnu[1] - Bnu[0]) / dtau_uw;
    } else if (bc.type == CALLABLE) {
        int mu_idx = bc.idxs(mu, int(to_obs));
        if (mu_idx == -1) {
            printf("Error in boundary condition indexing");
            assert(false);
        }
        Iupw = bc.bcData(la, mu_idx, 0);
    }

    pragmatic_1_fvm_1d_impl(fd, z_mu, to_obs, Iupw);
}

extern "C" {
    FormalSolver fs_provider() {
        return FormalSolver{pragmatic_1_fvm_1d, 1, 1, "pragmatic_1_fvm_1d"};
    }
}