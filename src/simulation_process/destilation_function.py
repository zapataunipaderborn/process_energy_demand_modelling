class DistillationEnergyDemand:
    def __init__(self, mass_flow, fraction_lowbp, fraction_highbp, T_inlet, T_boil, h_vap,
                 cp_distillate, cp_bottoms, T_outlet):
        self.mass_flow = mass_flow          # kg/s
        self.fraction_lowbp = fraction_lowbp
        self.fraction_highbp = fraction_highbp
        self.T_inlet = T_inlet
        self.T_boil = T_boil
        self.h_vap = h_vap                  # kJ/kg
        self.cp_distillate = cp_distillate  # kJ/kgK
        self.cp_bottoms = cp_bottoms        # kJ/kgK
        self.T_outlet = T_outlet

    def heating_demand(self):
        cp_feed = self.fraction_lowbp * self.cp_distillate + self.fraction_highbp * self.cp_bottoms
        Q_heat_feed = self.mass_flow * cp_feed * (self.T_boil - self.T_inlet)
        Q_vap = self.mass_flow * self.fraction_lowbp * self.h_vap
        return Q_heat_feed + Q_vap

    def cooling_demand(self):
        Q_cool_distillate = self.mass_flow * self.fraction_lowbp * (self.h_vap + self.cp_distillate * (self.T_boil - self.T_outlet))
        Q_cool_bottoms = self.mass_flow * self.fraction_highbp * self.cp_bottoms * (self.T_boil - self.T_outlet)
        return Q_cool_distillate + Q_cool_bottoms

    def energy_summary(self):
        return {
            "Heating_demand_kW": self.heating_demand(),
            "Cooling_demand_kW": self.cooling_demand(),
            "Temperatures_C": {
                "T_inlet": self.T_inlet,
                "T_boil": self.T_boil,
                "T_outlet": self.T_outlet
            }
        }


if __name__ == '__main__':
    distillation = DistillationEnergyDemand(
        mass_flow=12000/3600,
        fraction_lowbp=0.25,
        fraction_highbp=0.75,
        T_inlet=25,
        T_boil=137,
        h_vap=593,
        cp_distillate=2.52,
        cp_bottoms=2.38,
        T_outlet=30
    )
    result = distillation.energy_summary()
    print(result)