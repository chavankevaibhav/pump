import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize, root_scalar
import math

# --- PumpAnalysis Class ---
class PumpAnalysis:
    def __init__(self):
        # Pump characteristics (can be loaded from file/database in real application)
        # Added NPSHr data (required NPSH)
        self.pump_curves = {
            'Pump1': {
                'flow_rate': np.array([0, 10, 20, 30, 40, 50]),  # m³/h
                'head': np.array([50, 48, 45, 40, 33, 25]),      # meters
                'efficiency': np.array([0, 45, 65, 70, 65, 50]), # %
                'npshr': np.array([2, 2.5, 3, 4, 5.5, 7.5])       # meters
            },
            'Pump2': {
                'flow_rate': np.array([0, 15, 30, 45, 60, 75]),
                'head': np.array([60, 58, 54, 48, 40, 30]),
                'efficiency': np.array([0, 40, 60, 75, 70, 55]),
                'npshr': np.array([2.5, 3, 4, 5.5, 7, 9])
            }
            # Add more pumps here if needed
        }

        # Fluid properties (water by default)
        # Added vapor pressure
        self.fluid = {
            'density': 1000.0,    # kg/m³ - Changed to float
            'viscosity': 0.001, # Pa·s (dynamic viscosity)
            'vapor_pressure': 2340.0 # Pa (at 20°C for water) - Changed to float
        }

        # System properties
        # Added pipe roughness, atmospheric pressure, suction static level, suction pipe details
        # Added minor loss K-factors
        self.system = {
            'static_head_discharge': 10.0, # meters - Changed to float
            'static_head_suction': -2.0,   # meters - Changed to float
            'pipe_diameter_discharge': 0.1, # meters
            'pipe_length_discharge': 80.0, # meters - Changed to float
            'pipe_diameter_suction': 0.1, # meters
            'pipe_length_suction': 20.0,   # meters - Changed to float
            'pipe_roughness': 0.00005, # meters
            'atmospheric_pressure': 101325.0, # Pa - Changed to float
            'minor_losses_discharge': { # Dictionary of fitting type and quantity on discharge side
                'elbow_std': 4,
                'gate_valve_full_open': 1
            },
             'minor_losses_suction': { # Dictionary of fitting type and quantity on suction side
                'entrance_sharp': 1,
                'elbow_std': 2
            },
            'fitting_k_factors': { # Typical K-factors (can be expanded)
                'elbow_std': 0.9,
                'gate_valve_full_open': 0.2,
                'entrance_sharp': 0.5,
                'exit': 1.0 # Exit loss is often included as a minor loss
            }
        }

        # Gravity
        self.g = 9.81 # m/s²

    def convert_flow_rate(self, flow_rate_m3h, to_unit='m3/s'):
        """Converts flow rate between m³/h and m³/s"""
        if to_unit == 'm3/s':
            return np.array(flow_rate_m3h) / 3600.0 # Ensure float division
        elif to_unit == 'm3/h':
            return np.array(flow_rate_m3h) * 3600.0 # Ensure float multiplication
        else:
            raise ValueError("Invalid unit. Use 'm3/s' or 'm3/h'.")

    def calculate_velocity(self, flow_rate_m3h, diameter):
        """Calculate fluid velocity in pipe"""
        flow_rate_m3s = self.convert_flow_rate(flow_rate_m3h, 'm3/s')
        area = np.pi * (diameter/2.0)**2 # Ensure float division
        if area <= 0: return 0.0
        return flow_rate_m3s / area

    def calculate_reynolds_number(self, flow_rate_m3h, diameter):
        """Calculate Reynolds number"""
        velocity = self.calculate_velocity(flow_rate_m3h, diameter)
        if diameter <= 0 or self.fluid['viscosity'] <= 0: return 0.0
        Re = (self.fluid['density'] * velocity * diameter) / self.fluid['viscosity']
        return Re

    def calculate_friction_factor_haaland(self, Re, roughness, diameter):
        """Calculate friction factor using Haaland approximation"""
        if Re <= 0: return 0.0 # No flow, no friction
        if Re < 2000: # Laminar flow
            return 64.0 / Re # Ensure float division
        else: # Turbulent flow (Haaland approximation)
            e_D = roughness / diameter # Relative roughness
            # Ensure the term inside log10 is positive
            term1 = e_D / 3.7
            term2 = 6.9 / Re
            # Handle potential issues if term1 + term2 is zero or negative, though unlikely with positive inputs
            if term1 + term2 <= 0:
                 return 0.05 # Return a reasonable default or raise error
            log_term = np.log10(term1 + term2)
            # Handle potential division by zero if log_term is zero
            if log_term == 0:
                 return 0.05 # Return a reasonable default or raise error
            return (-1.8 * log_term)**-2


    def calculate_pipe_friction_loss(self, flow_rate_m3h, length, diameter, roughness):
        """Calculate friction head loss in a pipe section (major losses)"""
        flow_rate_m3s = self.convert_flow_rate(flow_rate_m3h, 'm3/s')
        if flow_rate_m3s <= 0 or diameter <= 0 or length <= 0 or self.g <= 0: return 0.0

        velocity = self.calculate_velocity(flow_rate_m3h, diameter)
        Re = self.calculate_reynolds_number(flow_rate_m3h, diameter)
        f = self.calculate_friction_factor_haaland(Re, roughness, diameter)

        friction_loss = (f * (length / diameter) * (velocity**2) / (2.0 * self.g)) # Ensure float division
        return friction_loss

    def calculate_minor_losses(self, flow_rate_m3h, diameter, minor_loss_dict):
        """Calculate minor head losses due to fittings"""
        if flow_rate_m3h <= 0 or diameter <= 0 or self.g <= 0: return 0.0

        velocity = self.calculate_velocity(flow_rate_m3h, diameter)
        velocity_head = (velocity**2) / (2.0 * self.g) # Ensure float division
        total_minor_loss_k = 0.0

        for fitting_type, quantity in minor_loss_dict.items():
            k_factor = self.system['fitting_k_factors'].get(fitting_type, 0.0) # Get K-factor, default to 0.0 if not found
            total_minor_loss_k += k_factor * quantity

        minor_loss = total_minor_loss_k * velocity_head
        return minor_loss

    def calculate_system_curve(self, flow_rate_m3h):
        """Calculate total system head requirement at given flow rate including minor losses"""
        # Suction side friction loss (major)
        friction_loss_suction = self.calculate_pipe_friction_loss(
            flow_rate_m3h,
            self.system['pipe_length_suction'],
            self.system['pipe_diameter_suction'],
            self.system['pipe_roughness']
        )

        # Discharge side friction loss (major)
        friction_loss_discharge = self.calculate_pipe_friction_loss(
            flow_rate_m3h,
            self.system['pipe_length_discharge'],
            self.system['pipe_diameter_discharge'],
            self.system['pipe_roughness']
        )

        # Suction side minor losses
        minor_loss_suction = self.calculate_minor_losses(
             flow_rate_m3h,
             self.system['pipe_diameter_suction'],
             self.system['minor_losses_suction']
        )

        # Discharge side minor losses
        minor_loss_discharge = self.calculate_minor_losses(
             flow_rate_m3h,
             self.system['pipe_diameter_discharge'],
             self.system['minor_losses_discharge']
        )


        total_friction_loss = friction_loss_suction + friction_loss_discharge + minor_loss_suction + minor_loss_discharge

        # Total system head is static head difference plus total losses
        total_static_head = self.system['static_head_discharge'] - self.system['static_head_suction']

        return total_static_head + total_friction_loss

    def calculate_npsha(self, flow_rate_m3h):
        """Calculate Net Positive Suction Head Available (NPSHa)"""
        if self.fluid['density'] <= 0 or self.g <= 0: return 0.0 # Avoid division by zero

        # Atmospheric head
        atmospheric_head = self.system['atmospheric_pressure'] / (self.fluid['density'] * self.g)

        # Vapor pressure head
        vapor_pressure_head = self.fluid['vapor_pressure'] / (self.fluid['density'] * self.g)

        # Suction side total head loss (major + minor)
        friction_loss_suction = self.calculate_pipe_friction_loss(
            flow_rate_m3h,
            self.system['pipe_length_suction'],
            self.system['pipe_diameter_suction'],
            self.system['pipe_roughness']
        )
        minor_loss_suction = self.calculate_minor_losses(
             flow_rate_m3h,
             self.system['pipe_diameter_suction'],
             self.system['minor_losses_suction']
        )
        total_suction_loss = friction_loss_suction + minor_loss_suction

        # Static suction head/lift (negative for lift, positive for head)
        # The system['static_head_suction'] is defined from pump centerline to liquid level.
        # NPSHa formula uses suction *lift* (positive if level is *below* centerline).
        # So, if static_head_suction is negative (lift), suction_lift = -self.system['static_head_suction'].
        # If static_head_suction is positive (submerged), suction_lift = -self.system['static_head_suction'].
        suction_lift = -self.system['static_head_suction']

        # NPSHa = Ha - Hv - Hfs - Hz (where Hz is static lift)
        npsha = atmospheric_head - vapor_pressure_head - total_suction_loss - suction_lift
        return npsha

    def combine_pumps_series(self, pump_names):
        """Combine head and efficiency curves for pumps in series"""
        if not pump_names:
            return None

        # Find the minimum and maximum flow rates across all selected pumps
        min_flow = max(min(self.pump_curves[name]['flow_rate']) for name in pump_names if name in self.pump_curves)
        max_flow = min(max(self.pump_curves[name]['flow_rate']) for name in pump_names if name in self.pump_curves)

        # Generate a common flow range for interpolation
        # Choose a step size, or number of points. Let's use a fixed number of points for smoothness.
        num_points = 100
        combined_flow_range = np.linspace(min_flow, max_flow, num_points)

        combined_head = np.zeros(num_points)
        combined_efficiency_sum = np.zeros(num_points) # Sum of (Eff * Head) for weighted average
        total_head_sum = np.zeros(num_points) # Sum of Head for weighted average

        for name in pump_names:
            if name in self.pump_curves:
                pump = self.pump_curves[name]
                # Create interpolation functions for each pump
                head_interp = interp1d(pump['flow_rate'], pump['head'], kind='cubic', bounds_error=False, fill_value=0.0)
                eff_interp = interp1d(pump['flow_rate'], pump['efficiency'], kind='cubic', bounds_error=False, fill_value=0.0)

                # For series, heads add up at the same flow rate
                head_at_flow = head_interp(combined_flow_range)
                combined_head += head_at_flow

                # For efficiency in series, if power is proportional to head,
                # the combined efficiency is approx (Total Head) / sum(Head_i / Eff_i)
                # A simpler approximation is a weighted average based on head
                eff_at_flow = eff_interp(combined_flow_range)
                combined_efficiency_sum += (eff_at_flow / 100.0) * head_at_flow # Convert efficiency to decimal
                total_head_sum += head_at_flow

        # Calculate weighted average efficiency
        # Avoid division by zero where total_head_sum is zero (usually at 0 flow)
        combined_efficiency = np.where(total_head_sum > 0, (combined_efficiency_sum / total_head_sum) * 100.0, 0.0)


        # NPSHr for series pumps: the NPSHr of the *first* pump in series matters most.
        # This is a simplification; the actual NPSHr depends on the specific arrangement.
        # For simplicity here, we'll take the maximum NPSHr among the selected pumps at each flow.
        combined_npshr = np.zeros(num_points)
        for name in pump_names:
             if name in self.pump_curves and 'npshr' in self.pump_curves[name]:
                 pump = self.pump_curves[name]
                 npshr_interp = interp1d(pump['flow_rate'], pump['npshr'], kind='cubic', bounds_error=False, fill_value=0.0)
                 npshr_at_flow = npshr_interp(combined_flow_range)
                 combined_npshr = np.maximum(combined_npshr, npshr_at_flow) # Take the maximum NPSHr at each flow

        # Clean up negative values that might result from interpolation extrapolation
        combined_head[combined_head < 0] = 0.0
        combined_efficiency[combined_efficiency < 0] = 0.0
        combined_npshr[combined_npshr < 0] = 0.0


        return {
            'flow_rate': combined_flow_range,
            'head': combined_head,
            'efficiency': combined_efficiency,
            'npshr': combined_npshr,
            'name': f"{'+'.join(pump_names)} (Series)"
        }


    def combine_pumps_parallel(self, pump_names):
        """Combine flow rate and efficiency curves for pumps in parallel"""
        if not pump_names:
            return None

        # Find the minimum and maximum head values across all selected pumps
        min_head = min(min(self.pump_curves[name]['head']) for name in pump_names if name in self.pump_curves)
        max_head = max(max(self.pump_curves[name]['head']) for name in self.pump_curves)

        # Generate a common head range for interpolation
        # Choose a step size, or number of points.
        num_points = 100
        combined_head_range = np.linspace(max_head, min_head, num_points) # Go from max head down to min head

        combined_flow = np.zeros(num_points)
        combined_power_sum = np.zeros(num_points) # Sum of power for combined efficiency calculation
        combined_flow_sum = np.zeros(num_points) # Sum of flow for combined efficiency calculation


        for name in pump_names:
            if name in self.pump_curves:
                pump = self.pump_curves[name]
                # Create interpolation functions for each pump (Flow as function of Head)
                # Need to ensure head values are strictly increasing/decreasing for interpolation
                # Sort by head and handle potential duplicate head values by averaging flow or taking first/last
                sorted_indices = np.argsort(pump['head'])
                sorted_head = pump['head'][sorted_indices]
                sorted_flow = pump['flow_rate'][sorted_indices]

                # Handle duplicate head values: average flow for duplicate heads
                unique_head, unique_indices = np.unique(sorted_head, return_index=True)
                # If there are duplicates, this simple unique might not be sufficient.
                # A more robust approach for parallel interpolation:
                # For each desired 'combined_head_range' point, find the flow from each pump's head curve.
                # This avoids interpolating flow vs head directly if the head curve isn't monotonic.

                # Let's use interpolation of Head vs Flow, and then find Flow for a given Head.
                head_interp = interp1d(pump['flow_rate'], pump['head'], kind='cubic', bounds_error=False, fill_value=0.0)
                eff_interp = interp1d(pump['flow_rate'], pump['efficiency'], kind='cubic', bounds_error=False, fill_value=0.0)
                npshr_interp = interp1d(pump['flow_rate'], pump['npshr'], kind='cubic', bounds_error=False, fill_value=0.0)


                # For parallel, flow rates add up at the same head
                # Need to find the flow rate for each pump at each head in combined_head_range
                flow_at_head = np.zeros(num_points)
                eff_at_head = np.zeros(num_points)
                npshr_at_head = np.zeros(num_points)

                for i, head_val in enumerate(combined_head_range):
                    # Find the flow rate where the pump's head curve equals head_val
                    # This requires solving head_interp(q) = head_val for q
                    def head_objective(flow_rate):
                        # Ensure flow rate is within pump's range for interpolation
                        flow_rate = np.clip(flow_rate, min(pump['flow_rate']), max(pump['flow_rate']))
                        return head_interp(flow_rate) - head_val

                    # Use root finding to find the flow rate for the given head
                    # Search within the pump's flow rate range
                    try:
                        # Find where the head difference is zero
                        sol = root_scalar(head_objective, bracket=[min(pump['flow_rate']), max(pump['flow_rate'])])
                        if sol.converged:
                            flow_at_head[i] = sol.root
                            # Get efficiency and NPSHr at this flow rate
                            eff_at_head[i] = eff_interp(sol.root)
                            npshr_at_head[i] = npshr_interp(sol.root)
                        else:
                            # If root finding fails, try to find the flow rate that minimizes the head difference
                            res = minimize(lambda q: abs(head_objective(q)), x0=np.mean(pump['flow_rate']),
                                           bounds=[(min(pump['flow_rate']), max(pump['flow_rate']))])
                            if res.success:
                                flow_at_head[i] = res.x[0]
                                eff_at_head[i] = eff_interp(res.x[0])
                                npshr_at_head[i] = npshr_interp(res.x[0])
                            else:
                                # If both fail, flow is 0 at this head for this pump
                                flow_at_head[i] = 0.0
                                eff_at_head[i] = 0.0
                                npshr_at_head[i] = 0.0

                    except Exception as e:
                         # Handle cases where interpolation might fail (e.g., head_val outside pump's head range)
                         # If head_val is above the pump's max head, flow is 0. If below min head, flow is max flow.
                         # This is a simplification, a proper check against the pump's head range is better.
                         # For now, if root finding fails, assume 0 flow.
                         flow_at_head[i] = 0.0
                         eff_at_head[i] = 0.0
                         npshr_at_head[i] = 0.0


                combined_flow += flow_at_head

                # For efficiency in parallel, the combined efficiency is approx (Total Flow * Head) / sum(Flow_i * Head / Eff_i)
                # This is equivalent to (Total Flow * Head) / sum(Power_i)
                # Power_i = (density * g * Flow_i * Head) / Eff_i
                # Combined Efficiency = (density * g * Sum(Flow_i) * Head) / Sum((density * g * Flow_i * Head) / Eff_i)
                # Combined Efficiency = Sum(Flow_i) / Sum(Flow_i / Eff_i)
                # Need to calculate power for each pump at this head
                power_at_head = self.calculate_power(flow_at_head, head_val, eff_at_head) # This function expects arrays
                combined_power_sum += power_at_head
                combined_flow_sum += flow_at_head


        # Calculate combined efficiency
        # Avoid division by zero where combined_power_sum is zero
        # Combined Power = Sum(Power_i)
        # Combined Efficiency = (density * g * Total Flow * Head) / Combined Power
        # Combined Efficiency = (density * g * combined_flow * combined_head_range) / combined_power_sum
        # Need to handle cases where combined_power_sum is 0
        combined_efficiency = np.where(combined_power_sum > 0,
                                       (self.fluid['density'] * self.g * (self.convert_flow_rate(combined_flow, 'm3/s')) * combined_head_range) / combined_power_sum * 100.0,
                                       0.0)

        # NPSHr for parallel pumps: Each pump needs its required NPSHr met based on its *individual* flow rate.
        # However, the system NPSHa is the same for all pumps at the suction header.
        # For simplicity here, we'll take the maximum NPSHr among the selected pumps at each head.
        combined_npshr = np.zeros(num_points)
        for name in pump_names:
             if name in self.pump_curves and 'npshr' in self.pump_curves[name]:
                 pump = self.pump_curves[name]
                 npshr_interp = interp1d(pump['flow_rate'], pump['npshr'], kind='cubic', bounds_error=False, fill_value=0.0)
                 # Need NPSHr at the flow rate *this pump* provides at the given combined head
                 npshr_at_head_this_pump = np.zeros(num_points)
                 for i, head_val in enumerate(combined_head_range):
                     def head_objective(flow_rate):
                         flow_rate = np.clip(flow_rate, min(pump['flow_rate']), max(pump['flow_rate']))
                         return head_interp(flow_rate) - head_val
                     try:
                        sol = root_scalar(head_objective, bracket=[min(pump['flow_rate']), max(pump['flow_rate'])])
                        if sol.converged:
                            npshr_at_head_this_pump[i] = npshr_interp(sol.root)
                        else:
                            res = minimize(lambda q: abs(head_objective(q)), x0=np.mean(pump['flow_rate']),
                                           bounds=[(min(pump['flow_rate']), max(pump['flow_rate']))])
                            if res.success:
                                npshr_at_head_this_pump[i] = npshr_interp(res.x[0])
                            else:
                                npshr_at_head_this_pump[i] = 0.0
                     except:
                         npshr_at_head_this_pump[i] = 0.0

                 combined_npshr = np.maximum(combined_npshr, npshr_at_head_this_pump) # Take the maximum NPSHr at each head


        # Sort combined curve by flow rate in ascending order
        sorted_indices = np.argsort(combined_flow)
        combined_flow = combined_flow[sorted_indices]
        combined_head = combined_head_range[sorted_indices] # Head range was already sorted descending, need to sort with flow
        combined_efficiency = combined_efficiency[sorted_indices]
        combined_npshr = combined_npshr[sorted_indices]

        # Clean up negative values that might result from interpolation extrapolation
        combined_flow[combined_flow < 0] = 0.0
        combined_head[combined_head < 0] = 0.0
        combined_efficiency[combined_efficiency < 0] = 0.0
        combined_npshr[combined_npshr < 0] = 0.0


        return {
            'flow_rate': combined_flow,
            'head': combined_head,
            'efficiency': combined_efficiency,
            'npshr': combined_npshr,
            'name': f"{'+'.join(pump_names)} (Parallel)"
        }


    def calculate_operating_point(self, pump_info):
        """
        Find operating point where pump curve intersects system curve.
        pump_info can be a pump name (str) or a combined pump dictionary.
        """
        if isinstance(pump_info, str):
            if pump_info not in self.pump_curves:
                # Handle case where pump_info is a string but not a valid pump name
                return None
            pump = self.pump_curves[pump_info]
            pump_name = pump_info
        elif isinstance(pump_info, dict) and 'flow_rate' in pump_info and 'head' in pump_info:
             # Assume it's a combined pump dictionary
             pump = pump_info
             pump_name = pump_info.get('name', 'Combined Pump')
        else:
            # Handle invalid input type
            return None


        # Create interpolation functions for pump curves
        # Use 'cubic' for smoother curves, ensure bounds_error=False and appropriate fill_value
        # Extrapolate with the last value to avoid sudden drops to 0 outside the defined range
        # Handle cases where the pump curve data might be empty or have only one point
        if pump['flow_rate'].size < 2:
             st.warning(f"Pump '{pump_name}' does not have enough data points for interpolation.")
             return None # Cannot interpolate with less than 2 points

        head_curve_interp = interp1d(pump['flow_rate'], pump['head'],
                                     kind='cubic', bounds_error=False, fill_value=(float(pump['head'][0]), float(pump['head'][-1])))

        # Handle cases where efficiency or NPSHr might not be available in the dict (e.g., for combined curves)
        eff_data = pump.get('efficiency', np.zeros_like(pump['flow_rate'], dtype=float)) # Ensure float type
        npshr_data = pump.get('npshr', np.zeros_like(pump['flow_rate'], dtype=float)) # Ensure float type

        eff_curve_interp = interp1d(pump['flow_rate'], eff_data,
                                    kind='cubic', bounds_error=False, fill_value=(float(eff_data[0]), float(eff_data[-1])) if eff_data.size > 0 else (0.0,0.0))
        npshr_curve_interp = interp1d(pump['flow_rate'], npshr_data,
                                      kind='cubic', bounds_error=False, fill_value=(float(npshr_data[0]), float(npshr_data[-1])) if npshr_data.size > 0 else (0.0,0.0))


        def head_difference(flow_rate):
            """Function to minimize: difference between pump head and system head"""
            # Ensure flow_rate is within reasonable bounds or interpolation is handled well
            # Use clip to keep flow within the pump's defined range for interpolation
            flow_rate = np.clip(flow_rate, float(min(pump['flow_rate'])), float(max(pump['flow_rate']))) # Ensure bounds are floats
            return head_curve_interp(flow_rate) - self.calculate_system_curve(flow_rate)

        # Find the root (where head_difference is zero)
        # Use a root finder instead of minimizing the absolute difference, it's often more robust
        try:
            # Define a search interval for the root (flow rate)
            # A reasonable interval is the range of flow rates for the pump curve
            flow_min, flow_max = float(min(pump['flow_rate'])), float(max(pump['flow_rate'])) # Ensure bounds are floats

            # Check if the system curve intersects the pump curve within the range
            # Evaluate head difference at min and max flow
            diff_min = head_difference(flow_min)
            diff_max = head_difference(flow_max)

            # If signs are different, a root exists. If signs are same, they might not intersect or touch.
            if np.sign(diff_min) != np.sign(diff_max):
                 # A root exists between flow_min and flow_max
                 sol = root_scalar(head_difference, bracket=[flow_min, flow_max])
                 if not sol.converged:
                      # If root finder failed, fall back to minimization
                      st.warning(f"Root finder failed to converge for {pump_name}. Minimizing absolute difference instead.")
                      def objective(flow_rate):
                         return abs(head_difference(flow_rate))
                      result = minimize(objective, x0=float(np.mean(pump['flow_rate'])), # Ensure initial guess is float
                                        bounds=[(flow_min, flow_max)])
                      if not result.success:
                         st.error(f"Failed to find operating point for {pump_name} using minimization (fallback).")
                         return None
                      operating_flow = result.x[0]
                 else:
                    operating_flow = sol.root
            else:
                 # No simple root found in the bracket. Find the point closest to zero difference.
                 st.warning(f"Root finder did not find a simple intersection for {pump_name}. Minimizing absolute difference instead.")
                 def objective(flow_rate):
                     return abs(head_difference(flow_rate))
                 result = minimize(objective, x0=float(np.mean(pump['flow_rate'])), # Ensure initial guess is float
                                   bounds=[(flow_min, flow_max)])
                 if not result.success:
                     st.error(f"Failed to find operating point for {pump_name} using minimization.")
                     return None
                 operating_flow = result.x[0]


            # Ensure operating flow is within interpolation range (or bounds if using clip)
            operating_flow = np.clip(operating_flow, float(min(pump['flow_rate'])), float(max(pump['flow_rate']))) # Ensure bounds are floats

            operating_head = head_curve_interp(operating_flow)
            operating_eff = eff_curve_interp(operating_flow)
            operating_npshr = npshr_curve_interp(operating_flow)
            operating_npsha = self.calculate_npsha(operating_flow)
            operating_power = self.calculate_power(operating_flow, operating_head, operating_eff)

            # Calculate additional metrics for detailed report
            flow_rate_m3s = self.convert_flow_rate(operating_flow, 'm3/s')
            velocity_discharge = self.calculate_velocity(operating_flow, self.system['pipe_diameter_discharge'])
            velocity_suction = self.calculate_velocity(operating_flow, self.system['pipe_diameter_suction'])
            reynolds_discharge = self.calculate_reynolds_number(operating_flow, self.system['pipe_diameter_discharge'])
            reynolds_suction = self.calculate_reynolds_number(operating_flow, self.system['pipe_diameter_suction'])
            friction_loss_discharge = self.calculate_pipe_friction_loss(operating_flow, self.system['pipe_length_discharge'], self.system['pipe_diameter_discharge'], self.system['pipe_roughness'])
            friction_loss_suction = self.calculate_pipe_friction_loss(operating_flow, self.system['pipe_length_suction'], self.system['pipe_diameter_suction'], self.system['pipe_roughness'])
            minor_loss_discharge = self.calculate_minor_losses(operating_flow, self.system['pipe_diameter_discharge'], self.system['minor_losses_discharge'])
            minor_loss_suction = self.calculate_minor_losses(operating_flow, self.system['pipe_diameter_suction'], self.system['minor_losses_suction'])

            npsh_margin = operating_npsha - operating_npshr
            cavitation_risk = "High Risk" if npsh_margin < 0 else ("Low Risk" if npsh_margin > 0.5 else "Moderate Risk") # Simple margin check

            return {
                'pump_name': pump_name,
                'flow_rate_m3h': float(operating_flow), # Ensure float for easy printing
                'head_m': float(operating_head),
                'efficiency_pct': float(operating_eff),
                'power_kW': float(operating_power),
                'npsha_m': float(operating_npsha),
                'npshr_m': float(operating_npshr),
                'npsh_margin_m': float(npsh_margin),
                'cavitation_risk': cavitation_risk,
                'velocity_discharge_mps': float(velocity_discharge),
                'velocity_suction_mps': float(velocity_suction),
                'reynolds_discharge': float(reynolds_discharge),
                'reynolds_suction': float(reynolds_suction),
                'friction_loss_discharge_m': float(friction_loss_discharge),
                'friction_loss_suction_m': float(friction_loss_suction),
                'minor_loss_discharge_m': float(minor_loss_discharge),
                'minor_loss_suction_m': float(minor_loss_suction),
                'total_static_head_m': float(self.system['static_head_discharge'] - self.system['static_head_suction']) # Ensure float
            }

        except Exception as e:
             st.error(f"An error occurred while calculating operating point for {pump_name}: {e}")
             return None # Return None or an error indicator


    def calculate_power(self, flow_rate_m3h, head_m, efficiency_pct):
        """Calculate required pump power in kW"""
        # This function can handle scalar or array inputs for flow, head, and efficiency
        flow_rate_m3s = self.convert_flow_rate(np.array(flow_rate_m3h, dtype=float), 'm3/s') # Ensure numpy array of floats
        head_m = np.array(head_m, dtype=float) # Ensure numpy array of floats
        efficiency_pct = np.array(efficiency_pct, dtype=float) # Ensure numpy array of floats

        # Avoid division by zero or negative efficiency
        efficiency_decimal = np.where(efficiency_pct > 0, efficiency_pct / 100.0, np.inf) # Ensure float division

        # Power (W) = density * g * flow_rate (m³/s) * head (m) / efficiency (decimal)
        # Handle cases where efficiency_decimal is inf (efficiency is 0 or negative)
        power_watt = np.where(efficiency_decimal < np.inf,
                              (self.fluid['density'] * self.g * flow_rate_m3s * head_m) / efficiency_decimal,
                              np.inf)

        # Ensure power is non-negative
        power_watt = np.maximum(power_watt, 0.0) # Ensure float

        return power_watt / 1000.0 # Convert Watts to kW - Ensure float division


    def optimize_pump_selection(self, required_flow_m3h):
        """Find best pump based on operating point proximity to required flow and efficiency"""
        best_pump_info = None
        # Initialize with a large difference to ensure the first valid pump is selected
        min_flow_diff = float('inf')
        best_op_point = None

        results = {}

        # Analyze individual pumps
        st.subheader("Analyzing Individual Pumps for Optimization")
        for pump_name in self.pump_curves:
            st.write(f"Analyzing {pump_name}...")
            op_point = self.calculate_operating_point(pump_name)
            if op_point:
                results[pump_name] = op_point

                # Calculate how close this pump's operating flow is to the required flow
                flow_diff = abs(op_point['flow_rate_m3h'] - required_flow_m3h)

                # We want the pump whose operating point flow is closest to the required flow.
                # You could add other criteria here, like must meet minimum head,
                # or only consider if efficiency is above a threshold.
                # For now, let's just find the one closest in flow.
                if flow_diff < min_flow_diff:
                    min_flow_diff = flow_diff
                    best_pump_info = pump_name
                    best_op_point = op_point
                st.write(f"  Operating Flow: {op_point['flow_rate_m3h']:.2f} m³/h (Difference: {flow_diff:.2f} m³/h)")
            else:
                st.write(f"  Could not calculate operating point for {pump_name}. Skipping.")


        # Note: This optimization currently only considers individual pumps.
        # Extending it to consider all possible series/parallel combinations
        # would require generating all combinations and analyzing each, which can be computationally intensive.
        # For this example, we'll stick to individual pump optimization.


        return {
            'required_flow_m3h': required_flow_m3h,
            'best_pump': best_pump_info, # This is the name of the best individual pump
            'best_pump_operating_point': best_op_point, # Operating point of the best individual pump
            'all_operating_points': results # Operating points for all individual pumps
        }

    def plot_operating_point(self, pump_info):
        """
        Plot pump curve with system curve, operating point, and NPSH curves.
        pump_info can be a pump name (str) or a combined pump dictionary.
        """
        if isinstance(pump_info, str):
            if pump_info not in self.pump_curves:
                st.error(f"Pump '{pump_info}' not found in database.")
                return None
            pump = self.pump_curves[pump_info]
            pump_name = pump_info
        elif isinstance(pump_info, dict) and 'flow_rate' in pump_info and 'head' in pump_info:
             # Assume it's a combined pump dictionary
             pump = pump_info
             pump_name = pump_info.get('name', 'Combined Pump')
        else:
            st.error("Invalid input for plot_operating_point. Must be pump name or combined pump dict.")
            return None


        op_point = self.calculate_operating_point(pump_info)

        if not op_point:
            st.warning(f"Could not plot operating point for {pump_name} as calculation failed.")
            return None

        # Generate curves
        # Use the pump's flow rate range for plotting
        flow_min, flow_max = min(pump['flow_rate']), max(pump['flow_rate'])
        # Extend slightly beyond the range for better visualization if needed, but be mindful of extrapolation
        # Ensure the plot range includes the operating point flow rate
        plot_flow_min = min(flow_min * 0.8, op_point['flow_rate_m3h'] * 0.9)
        plot_flow_max = max(flow_max * 1.2, op_point['flow_rate_m3h'] * 1.1)
        plot_flow_range = np.linspace(plot_flow_min, plot_flow_max, 200)

        # Filter plot range to avoid potential issues with extreme extrapolation or negative flows
        plot_flow_range = plot_flow_range[(plot_flow_range >= 0)]


        # Interpolation functions (using cubic, ensure bounds_error=False)
        # Extrapolate with the last value to avoid sudden drops to 0 outside the defined range
        # Handle cases where the pump curve data might be empty or have only one point
        if pump['flow_rate'].size < 2:
             st.warning(f"Pump '{pump_name}' does not have enough data points for plotting.")
             return None # Cannot interpolate with less than 2 points

        head_curve_interp = interp1d(pump['flow_rate'], pump['head'],
                                     kind='cubic', bounds_error=False, fill_value=(float(pump['head'][0]), float(pump['head'][-1])))
        # Handle cases where efficiency or NPSHr might not be available in the dict (e.g., for combined curves)
        eff_data = pump.get('efficiency', np.zeros_like(pump['flow_rate'], dtype=float))
        npshr_data = pump.get('npshr', np.zeros_like(pump['flow_rate'], dtype=float))

        eff_curve_interp = interp1d(pump['flow_rate'], eff_data,
                                    kind='cubic', bounds_error=False, fill_value=(float(eff_data[0]), float(eff_data[-1])) if eff_data.size > 0 else (0.0,0.0))
        npshr_curve_interp = interp1d(pump['flow_rate'], npshr_data,
                                      kind='cubic', bounds_error=False, fill_value=(float(npshr_data[0]), float(npshr_data[-1])) if npshr_data.size > 0 else (0.0,0.0))


        pump_head = head_curve_interp(plot_flow_range)
        system_head = [self.calculate_system_curve(q) for q in plot_flow_range]
        npshr_plot = npshr_curve_interp(plot_flow_range)
        npsha_plot = [self.calculate_npsha(q) for q in plot_flow_range]

        # Create figures
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        # Plot Head and System Curves
        ax1.plot(plot_flow_range, pump_head, 'b-', label=f'{pump_name} Head Curve')
        ax1.plot(plot_flow_range, system_head, 'g-', label='System Curve')
        ax1.plot(op_point['flow_rate_m3h'], op_point['head_m'], 'ro', markersize=8, label='Operating Point')
        ax1.set_xlabel('Flow Rate (m³/h)')
        ax1.set_ylabel('Head (m)')
        ax1.set_title(f'Pump Operating Point - {pump_name}')
        ax1.grid(True)
        ax1.legend()
        ax1.set_ylim(bottom=0) # Start head axis from 0
        ax1.set_xlim(left=0) # Start flow axis from 0


        # Plot NPSH Curves
        ax2.plot(plot_flow_range, npsha_plot, 'm-', label='NPSH Available')
        ax2.plot(plot_flow_range, npshr_plot, 'c-', label=f'{pump_name} NPSH Required')
        ax2.plot(op_point['flow_rate_m3h'], op_point['npsha_m'], 'mo', markersize=8)
        ax2.plot(op_point['flow_rate_m3h'], op_point['npshr_m'], 'co', markersize=8)
        # Add text for NPSH Margin at operating point
        ax2.text(op_point['flow_rate_m3h'], op_point['npsha_m'],
                 f" NPSHa: {op_point['npsha_m']:.2f}m",
                 fontsize=9, verticalalignment='bottom')
        ax2.text(op_point['flow_rate_m3h'], op_point['npshr_m'],
                 f" NPSHr: {op_point['npshr_m']:.2f}m",
                 fontsize=9, verticalalignment='top')


        ax2.set_xlabel('Flow Rate (m³/h)')
        ax2.set_ylabel('NPSH (m)')
        ax2.set_title(f'NPSH Analysis - {pump_name}')
        ax2.grid(True)
        ax2.legend()
        ax2.set_ylim(bottom=0) # Start NPSH axis from 0
        ax2.set_xlim(left=0) # Start flow axis from 0


        plt.tight_layout()

        return op_point, fig1, fig2

    def report_operating_point(self, op_point):
        """Prints a detailed report of the operating point"""
        if not op_point:
            st.warning("No operating point data to report.")
            return

        st.subheader("Operating Point Details")
        st.write(f"**Pump:** {op_point['pump_name']}")
        st.write(f"**Flow Rate:** {op_point['flow_rate_m3h']:.2f} m³/h")
        st.write(f"**Head:** {op_point['head_m']:.2f} m")
        st.write(f"**Efficiency:** {op_point['efficiency_pct']:.1f} %")
        st.write(f"**Power:** {op_point['power_kW']:.2f} kW")
        st.markdown("---")
        st.write(f"**NPSH Available (NPSHa):** {op_point['npsha_m']:.2f} m")
        st.write(f"**NPSH Required (NPSHr):** {op_point['npshr_m']:.2f} m")
        st.write(f"**NPSH Margin (NPSHa - NPSHr):** {op_point['npsh_margin_m']:.2f} m")
        st.write(f"**Cavitation Risk:** {op_point['cavitation_risk']}")
        st.markdown("---")
        st.write(f"**Discharge Velocity:** {op_point['velocity_discharge_mps']:.2f} m/s")
        st.write(f"**Suction Velocity:** {op_point['velocity_suction_mps']:.2f} m/s")
        st.write(f"**Discharge Reynolds Number:** {op_point['reynolds_discharge']:.0f}")
        st.write(f"**Suction Reynolds Number:** {op_point['reynolds_suction']:.0f}")
        st.write(f"**Discharge Friction Loss (Major):** {op_point['friction_loss_discharge_m']:.2f} m")
        st.write(f"**Suction Friction Loss (Major):** {op_point['friction_loss_suction_m']:.2f} m")
        st.write(f"**Discharge Minor Loss:** {op_point['minor_loss_discharge_m']:.2f} m")
        st.write(f"**Suction Minor Loss:** {op_point['minor_loss_suction_m']:.2f} m")
        st.write(f"**Total Static Head:** {op_point['total_static_head_m']:.2f} m")
        st.markdown("---")


# --- Streamlit App Code ---

st.title("Pump and System Curve Analysis")

# Initialize PumpAnalysis object
# Use st.session_state to persist the object across reruns
if 'pump_system' not in st.session_state:
    st.session_state.pump_system = PumpAnalysis()

pump_system = st.session_state.pump_system

# --- Sidebar for Inputs ---
st.sidebar.header("System Parameters")

with st.sidebar.expander("Fluid Properties"):
    pump_system.fluid['density'] = st.number_input("Density (kg/m³)", value=pump_system.fluid['density'], min_value=1.0, format="%f") # Added format
    pump_system.fluid['viscosity'] = st.number_input("Viscosity (Pa·s)", value=pump_system.fluid['viscosity'], format="%e", min_value=0.0)
    pump_system.fluid['vapor_pressure'] = st.number_input("Vapor Pressure (Pa)", value=pump_system.fluid['vapor_pressure'], min_value=0.0, format="%f") # Added format

with st.sidebar.expander("Static Heads"):
    pump_system.system['static_head_discharge'] = st.number_input("Discharge Static Head (m)", value=pump_system.system['static_head_discharge'], format="%f") # Added format
    pump_system.system['static_head_suction'] = st.number_input("Suction Static Head (m)", value=pump_system.system['static_head_suction'], format="%f") # Added format

with st.sidebar.expander("Discharge Pipe"):
    pump_system.system['pipe_diameter_discharge'] = st.number_input("Diameter (m)", value=pump_system.system['pipe_diameter_discharge'], min_value=0.001, format="%f") # Added format
    pump_system.system['pipe_length_discharge'] = st.number_input("Length (m)", value=pump_system.system['pipe_length_discharge'], min_value=0.0, format="%f") # Added format
    pump_system.system['pipe_roughness'] = st.number_input("Roughness (m)", value=pump_system.system['pipe_roughness'], format="%e", min_value=0.0)

    st.subheader("Discharge Minor Losses (Quantity)")
    # Allow user to add/edit minor losses
    # This is a simplified way to handle dynamic dict input
    current_minor_losses_discharge = pump_system.system['minor_losses_discharge'].copy()
    fitting_types = list(pump_system.system['fitting_k_factors'].keys())
    selected_discharge_fittings = st.multiselect("Select Discharge Fittings", options=fitting_types, default=list(current_minor_losses_discharge.keys()))

    new_minor_losses_discharge = {}
    for fitting in selected_discharge_fittings:
        quantity = st.number_input(f"{fitting.replace('_', ' ').title()} Quantity", value=current_minor_losses_discharge.get(fitting, 0), min_value=0, format="%d")
        if quantity > 0:
            new_minor_losses_discharge[fitting] = quantity
    pump_system.system['minor_losses_discharge'] = new_minor_losses_discharge


with st.sidebar.expander("Suction Pipe"):
    pump_system.system['pipe_diameter_suction'] = st.number_input("Diameter (m) ", value=pump_system.system['pipe_diameter_suction'], min_value=0.001, format="%f") # Added format
    pump_system.system['pipe_length_suction'] = st.number_input("Length (m) ", value=pump_system.system['pipe_length_suction'], min_value=0.0, format="%f") # Added format
    # Suction pipe often has the same roughness as discharge, but can be made separate if needed
    # pump_system.system['pipe_roughness_suction'] = st.number_input("Roughness (m) ", value=pump_system.system['pipe_roughness'], format="%e", min_value=0.0)

    st.subheader("Suction Minor Losses (Quantity)")
    current_minor_losses_suction = pump_system.system['minor_losses_suction'].copy()
    selected_suction_fittings = st.multiselect("Select Suction Fittings", options=fitting_types, default=list(current_minor_losses_suction.keys()))

    new_minor_losses_suction = {}
    for fitting in selected_suction_fittings:
        quantity = st.number_input(f"{fitting.replace('_', ' ').title()} Quantity ", value=current_minor_losses_suction.get(fitting, 0), min_value=0, format="%d")
        if quantity > 0:
            new_minor_losses_suction[fitting] = quantity
    pump_system.system['minor_losses_suction'] = new_minor_losses_suction


with st.sidebar.expander("Fitting K-Factors"):
    st.write("Edit default K-factors here:")
    current_k_factors = pump_system.system['fitting_k_factors'].copy()
    new_k_factors = {}
    for fitting, k in current_k_factors.items():
         new_k_factors[fitting] = st.number_input(f"{fitting.replace('_', ' ').title()} K-Factor", value=k, min_value=0.0, format="%f")
    # Add option to add new fitting types? More complex input needed.
    pump_system.system['fitting_k_factors'] = new_k_factors


st.sidebar.header("Pump Selection")
pump_mode = st.sidebar.radio("Select Mode", ["Individual Pump", "Pumps in Series", "Pumps in Parallel"])

available_pumps = list(pump_system.pump_curves.keys())
selected_pump_info = None

if pump_mode == "Individual Pump":
    selected_pump_name = st.sidebar.selectbox("Select a Pump", available_pumps)
    if selected_pump_name:
        selected_pump_info = selected_pump_name # Pass the name to analysis functions
elif pump_mode == "Pumps in Series":
    selected_pump_names_series = st.sidebar.multiselect("Select Pumps for Series", available_pumps)
    if len(selected_pump_names_series) > 0:
        selected_pump_info = pump_system.combine_pumps_series(selected_pump_names_series)
        if selected_pump_info is None:
             st.sidebar.warning("Select at least one pump for series operation.")
elif pump_mode == "Pumps in Parallel":
    selected_pump_names_parallel = st.sidebar.multiselect("Select Pumps for Parallel", available_pumps)
    if len(selected_pump_names_parallel) > 0:
        selected_pump_info = pump_system.combine_pumps_parallel(selected_pump_names_parallel)
        if selected_pump_info is None:
             st.sidebar.warning("Select at least one pump for parallel operation.")


st.sidebar.header("Optimization")
required_flow_optimize = st.sidebar.number_input("Required Flow for Optimization (m³/h)", value=35.0, min_value=0.0, format="%f") # Added format
run_optimization_button = st.sidebar.button("Run Optimization")


# --- Main Area for Results ---
st.header("Analysis Results")

if selected_pump_info:
    st.subheader(f"Analyzing: {selected_pump_info.get('name', selected_pump_info) if isinstance(selected_pump_info, dict) else selected_pump_info}")

    if st.button("Calculate Operating Point"):
        op_point_result = pump_system.plot_operating_point(selected_pump_info)
        if op_point_result:
            op_point, fig_head, fig_npsh = op_point_result
            pump_system.report_operating_point(op_point)
            st.pyplot(fig_head)
            st.pyplot(fig_npsh)
            plt.close(fig_head) # Close figures to free memory
            plt.close(fig_npsh)
        else:
            st.error("Failed to calculate or plot the operating point.")

else:
    st.info("Please select a pump or combination from the sidebar to calculate the operating point.")

st.markdown("---")

if run_optimization_button:
    st.header("Optimization Results")
    optimization_results = pump_system.optimize_pump_selection(required_flow_optimize)

    st.write(f"**Required Flow:** {optimization_results['required_flow_m3h']:.2f} m³/h")
    if optimization_results['best_pump']:
        st.write(f"**Best Individual Pump (closest operating flow):** {optimization_results['best_pump']}")
        st.subheader("Operating point of the Best Individual Pump:")
        pump_system.report_operating_point(optimization_results['best_pump_operating_point'])
    else:
        st.warning("No suitable individual pump found among the options.")

    st.subheader("Operating Points for All Individual Pumps")
    if optimization_results['all_operating_points']:
        for pump_name, op_data in optimization_results['all_operating_points'].items():
            st.write(f"**{pump_name}:** Flow={op_data['flow_rate_m3h']:.2f} m³/h, "
                      f"Head={op_data['head_m']:.2f} m, "
                      f"Eff={op_data['efficiency_pct']:.1f}%, "
                      f"Power={op_data['power_kW']:.2f} kW, "
                      f"NPSHa={op_data['npsha_m']:.2f} m, "
                      f"NPSHr={op_data['npshr_m']:.2f} m, "
                      f"Cavitation Risk: {op_data['cavitation_risk']}")
    else:
        st.info("No operating points could be calculated for any individual pump.")

