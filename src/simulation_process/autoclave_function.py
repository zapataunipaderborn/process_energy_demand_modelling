import numpy as np
import CoolProp.CoolProp as CP
import pandas as pd
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

def sterilization_profile(t_ges = 6000,
    m_cp = 63032.0,  # [kJ/K]
    T_start = 317.15,
    T_steri = 394.25,
    T_amb = 293.15,
    T_final = 333.15,
    T_steam = 443.15,
    T_cool_in = 288.15,
    m_dot_steam = 1.94,
    m_dot_cool = 14.00,
    cp_water = 4.18,
    A_U = 0.20797,
    hold_duty_factor = 1.0,   # scales the HOLDING duty only (heat/cool untouched):
                              # 1.0 = vessel started cold and pays the full A_U loss,
                              # < 1.0 = still warm from the previous cycle. The caller
                              # derives it from the schedule; see generate_process_1.

    fill_factor = 0.9,
    delta_T_storage = 30,
    visual=False):

    t_heating = t_ges * 0.27
    t_holding = t_ges * 0.53
    t_cooling = t_ges * 0.19
    t_phase = round(0.55 * t_heating)  # heat demand starts tapering off earlier

    def get_latent_heat(T, fluid='Water'):
        h_liq = CP.PropsSI('H', 'T', T, 'Q', 0, fluid)
        h_vap = CP.PropsSI('H', 'T', T, 'Q', 1, fluid)
        latent_heat = h_vap - h_liq
        return latent_heat

    def get_steam_properties(T, fluid='Water'):
        h1_prime = CP.PropsSI('H', 'T', T, 'Q', 0, fluid)  # J/kg
        h1_double_prime = CP.PropsSI('H', 'T', T, 'Q', 1, fluid)  # J/kg
        rho_liq = CP.PropsSI('D', 'T', T, 'Q', 0, fluid)
        v1_prime = 1 / rho_liq
        return h1_prime, h1_double_prime, v1_prime

    def calculate_storage_volume(m_dampf, T1, fill_factor, delta_T, fluid='Water'):
        if not (0.9 <= fill_factor <= 0.95):
            raise ValueError("Fill factor must be between 0.9 and 0.95")
        h1_prime, h1_double_prime, v1_prime = get_steam_properties(T1, fluid)
        T2 = T1 - delta_T
        h2_prime, h2_double_prime, _ = get_steam_properties(T2, fluid)
        numerator = m_dampf
        denominator = (fill_factor / v1_prime) * (
                    (h1_prime - h2_prime) / (0.5 * (h1_double_prime + h2_double_prime) - h2_prime))
        V = numerator / denominator
        return V

    def calculate_steam_demand_holding(A_U, T_steri, T_amb, t_holding, h_vap_kJkg):
        Q_loss = A_U * (T_steri - T_amb)  # kW
        m_steam_holding = (Q_loss * t_holding) / h_vap_kJkg  # kg
        return m_steam_holding

    def calculate_cooling_demand(m_cp, T_steri, T_final):
        delta_T = T_steri - T_final
        if delta_T < 0:
            raise ValueError("Final temperature must be lower than sterilization temperature for cooling.")
        cooling_energy_kJ = m_cp * delta_T
        cooling_energy_kWh = cooling_energy_kJ / 3600
        return cooling_energy_kJ, cooling_energy_kWh

    # === Latent heat
    h_vap = get_latent_heat(T_steam)
    h_vap_kJkg = h_vap / 1000

    # === PHASE 1 (0 - t_phase) ===
    t1 = np.arange(0, int(t_phase) + 1)
    C1 = (T_start - T_amb) - (m_dot_steam * h_vap_kJkg / A_U)
    Q_dot = m_dot_steam * h_vap_kJkg
    T1 = ((Q_dot) / A_U + T_amb) + C1 * np.exp(-A_U / m_cp * t1)
    T1 = T1 + (T_start - T1[0])

    # === PHASE 2 (t_phase+1 - t_heating) ===
    t2 = np.arange(int(t_phase) + 1, int(t_heating) + 1)
    m_dot2 = m_dot_steam * (1 - (t2 - t_phase) / (t_heating - t_phase))
    T_inf = (m_dot2 * h_vap_kJkg) / A_U + T_amb
    delta_T = T_inf - T_steri
    exp_term = np.exp(-(t2 - t_phase) / (m_cp / A_U))
    T2 = T_inf - delta_T * exp_term
    T2 = np.minimum(T2, T_steri)
    T2 = T2 - (T2[0] - T1[-1]) * (1 - (t2 - t_phase)/(t_heating - t_phase))

    # === Gesamtdaten kombinieren
    t_total = np.concatenate([t1, t2])
    T_total = np.concatenate([T1, T2])
    # === PHASE 1 (t1): Konstanter Massenstrom
    Q_dot_steam1 = np.full_like(t1, m_dot_steam * h_vap_kJkg)  # [kW]
    # === PHASE 2 (t2): Abnehmender Massenstrom
    Q_dot_steam2 = m_dot2 * h_vap_kJkg  # [kW]
    # === Gesamter Leistungsvektor und Zeitvektor
    Q_dot_steam_total = np.concatenate([Q_dot_steam1, Q_dot_steam2])
    t_steam_total = np.concatenate([t1, t2])


    # === Dampfmassenberechnung ===
    m_dampf_phase1 = m_dot_steam * (t1[-1] - t1[0])
    m_dampf_phase2 = trapezoid(m_dot2, t2)
    m_dampf_gesamt = m_dampf_phase1 + m_dampf_phase2
    Q_heating = m_dampf_gesamt * h_vap_kJkg / 3600

    # Holding phase
    m_steam_holding = calculate_steam_demand_holding(A_U, T_steri, T_amb, t_holding,
                                                     h_vap_kJkg) * hold_duty_factor
    m_dampf_gesamt = m_dampf_gesamt + m_steam_holding

    # === Speicherberechnung ===
    volume_storage = calculate_storage_volume(m_dampf_gesamt, T_steam, fill_factor, delta_T_storage)

    # Ziel-Kühlenergie aus Systemmodell
    Q_target_kJ, _ = calculate_cooling_demand(m_cp, T_steri, T_final)
    Q_target_kWh = Q_target_kJ / 3600
    cp_cool = cp_water
    m_dot_cool_kg_s = m_dot_cool
    t3 = np.arange(int(t_heating) + int(t_holding) + 1, int(t_ges) + 1)
    t_cool_relative = t3 - t3[0]
    t_cool_duration = t_cool_relative[-1]
    # Iterative Suche nach passendem k
    k = 1e-3
    tolerance = 0.01
    max_iterations = 100
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        exp_term = 1 - np.exp(-k * t_cool_duration)
        delta_T0 = (Q_target_kJ * k) / (m_dot_cool_kg_s * cp_cool * exp_term)
        T0_out = T_cool_in + delta_T0
        if T0_out <= T_steri:
            break
        k *= 0.9
    # Temperaturverlauf: a smooth rise-then-fall "bump" instead of a pure exponential
    # decay — a quick but continuous ramp-up (no instant jump when cooling switches
    # on), a brief near-peak plateau, then an S-shaped decay back down. The peak is
    # fixed at delta_T0 (already guaranteed <= T_steri by the search above, so no
    # clipping is ever needed — that clipping was what produced flat-topped,
    # rectangle-looking curves before).
    tau_rise   = max(1.0, t_cool_duration / 25.0)   # quick, continuous ramp-up
    t_rise_mid = t_cool_duration * 0.06
    tau_fall   = max(1.0, t_cool_duration / 6.0)    # decay width relative to duration
    t_fall_mid = t_cool_duration * 0.40             # where the decay is centered

    rise = 1.0 / (1.0 + np.exp(-(t_cool_relative - t_rise_mid) / tau_rise))
    fall = 1.0 / (1.0 + np.exp((t_cool_relative - t_fall_mid) / tau_fall))
    sigma = rise * fall

    T_outlet_cool = T_cool_in + delta_T0 * sigma
    delta_T_cool = T_outlet_cool - T_cool_in
    Q_dot_cool = m_dot_cool_kg_s * cp_cool * (T_outlet_cool - T_cool_in)  # [kW]
    Q_kJ_check = trapezoid(delta_T_cool * m_dot_cool_kg_s * cp_cool, t3)
    Q_kWh_check = Q_kJ_check / 3600
    T_out_avg = trapezoid(T_outlet_cool, t3) / (t3[-1] - t3[0])


    # ===================== Q HOLDING calculation =====================
    # Option A (physical, equal to `A_U*(T_steri-T_amb)`):
    Q_holding_fill = hold_duty_factor * A_U*(T_steri-T_amb)  # [kW], same as Q_loss (heat loss compensated by holding steam)
    # Option B (based on used/averaged steam mass): can also use
    m_dot_holding = m_steam_holding / len(np.arange(int(t_heating)+1, int(t_heating)+int(t_holding)+1))
    Q_holding_avgsteam = m_dot_holding * h_vap_kJkg

    # ===================== Build result DataFrame ====================
    # Heating phase (already calculated)
    df_heating = pd.DataFrame({
        'time': t_total,
        'temperature_c': T_total-273.15,  # Convert K to C
        'phase': ['heat'] * len(t_total),
        'Q_kW': Q_dot_steam_total
    })

    # Holding phase (constant T, constant Q on average — with realistic steam-valve
    # chatter around that level, since a perfectly flat line looks unrealistic)
    t_holding_abs = np.arange(int(t_heating) + 1, int(t_heating) + int(t_holding) + 1)
    T_holding_arr = np.full_like(t_holding_abs, T_steri)
    # For Q: Option A is customary (total heat loss),
    # Option B is per above. Here, **A** (Q_holding_fill).
    holding_noise_rel = 0.06
    Q_holding_arr = Q_holding_fill + np.random.normal(
        0.0, Q_holding_fill * holding_noise_rel, len(t_holding_abs)
    )
    Q_holding_arr = np.clip(Q_holding_arr, 0.0, None)
    df_holding = pd.DataFrame({
        'time': t_holding_abs,
        'temperature_c': T_holding_arr-273.15,
        'phase': ['hold'] * len(t_holding_abs),
        'Q_kW': Q_holding_arr
    })

    # Cooling phase
    df_cooling = pd.DataFrame({
        'time': t3,
        'temperature_c': T_outlet_cool-273.15,
        'phase': ['cool'] * len(t3),
        'Q_kW': Q_dot_cool
    })

    results_df = pd.concat([df_heating, df_holding, df_cooling], ignore_index=True)

    results_df.reset_index(drop=True, inplace=True)
    results_df['time'] = results_df.index

    results_df = results_df.rename(columns={'temperature_c': 'temp', 'Q_kW': 'Q', 'phase': 'type'})

    if visual is True:

        df = results_df.copy()

        plt.figure(figsize=(12, 6))
        plt.plot(t_total, T_total, label="Temperature T(t)", color='blue')
        plt.axhline(y=T_steri, color='red', linestyle='--', label=f'target temperature {T_steri} K')
        plt.axvline(x=t_phase, color='gray', linestyle='--', label=f'phase change at t = {t_phase} s')
        plt.xlabel("Zeit in s")
        plt.ylabel("Temperatur in K")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    

        print(f"[✓] final k-value: {k:.6f}")
        print(f"start temperature outlet : {T0_out:.2f} K")
        print(f"average outlet temperature: {T_out_avg:.2f} K")
        print(f"cooling energy target    : {Q_target_kWh:.2f} kWh")
        print(f"cooling energy calculated: {Q_kWh_check:.2f} kWh")

        plt.figure(figsize=(10, 5))
        plt.plot(t_steam_total, Q_dot_steam_total, label='heating capacity', color='crimson')
        plt.xlabel("time in s")
        plt.ylabel("heating capacity in kW")
        plt.title("current heating capacity during the heating phase")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(t3, T_outlet_cool, label='cooling water outlet temperature', color='darkorange')
        plt.axhline(T_cool_in, linestyle='--', color='gray', label='cooling water inlet temperature')
        plt.axhline(T_steri, linestyle='--', color='red', label='sterilization temperature')
        plt.xlabel("time in s")
        plt.ylabel("temperature in K")
        plt.title("cooling water temperature profile")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(t3, Q_dot_cool, label='current cooling capacity', color='steelblue')
        plt.xlabel("time in s")
        plt.ylabel("cooling capacity in kW")
        plt.title("current cooling capacity during the cooling phase")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))

        for phase in df['type'].unique():
            mask = df['type'] == phase
            plt.plot(df['time'][mask], df['temp'][mask], label=phase)
        plt.xlabel("time in s")
        plt.ylabel("temperature in C")
        plt.title("Temperature profile")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))

        # Loop through each unique phase and plot separately
        for phase in df['type'].unique():
            mask = df['type'] == phase
            plt.plot(df['time'][mask], df['Q'][mask], label=phase)

        plt.xlabel("time in s")
        plt.ylabel("Q in kW") 
        plt.title("Q profile")
        plt.grid(True)
        plt.legend(title="Phase")
        plt.tight_layout()
        plt.show()

        print(f"Latent heat of vaporization at {T_steam} K: {h_vap_kJkg:.2f} kJ/kg")
        print(f"Steam demand during heating phase: {m_dampf_gesamt:.2f} kg")
        print(f"Energy demand: {Q_heating:.2f} kWh")
        print(f"Steam demand during holding phase: {m_steam_holding:.2f} kg")
        print(f"Storage volume: {volume_storage:.4f} m³")

    return results_df
