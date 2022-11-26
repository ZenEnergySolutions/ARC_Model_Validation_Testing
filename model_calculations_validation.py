import numpy as np
import pandas as pd
from bokeh.embed import components
from bokeh.io import curdoc
from bokeh.models import LinearAxis, Range1d
from bokeh.plotting import figure
from test_helper import Timer

curdoc().theme = "dark_minimal"


def create_csv(duty_cycle, results):
    """Create and save csv from the results of the model

      Args:
          duty_cycle (dataframe object): Main duty cycle dataframe (dc)
          results (dataframe object): Results dataframe
      """
    duty_cycle.to_csv('./routemodel/static/routemodel/csv/results_csv/model.csv', index=False)
    dc_sum = duty_cycle.sum()
    dc_sum.to_csv('./routemodel/static/routemodel/csv/results_csv/model_sum.csv')
    results.to_csv('./routemodel/static/routemodel/csv/results_csv/results.csv', index=False)
    results_stats = results.describe()
    results_stats.to_csv('./routemodel/static/routemodel/csv/results_csv/results_stats.csv')
    sum_table = results.sum()
    sum_table.loc['Cycle'] = duty_cycle['cycle'][0]
    sum_table.to_csv('./routemodel/static/routemodel/csv/results_csv/results_summary.csv')


def dc_forces(dc, gvw, GRAVITY, RHO_AIR, frontal_area, vehicle_cd, wheel_crr):
    """Calculate distance traveled, speed, time steps and forces.

      Args:
          dc (dataframe object): Main model dataframe
          gvw (float): Gross Vehicle Weight (kg)
          GRAVITY (float): Gravitational Constant (m/s2)
          RHO_AIR (float): Air Density Constant (kg/m3)
          frontal_area (float): Area of front of vehicle (m2)
          vehicle_cd (float): Drag Coefficient of vehicle
          wheel_crr (float): Rolling Wheel Coefficient of Vehicle

      Returns:
          dc (dataframe object):  Main model DF with added calculated fields
      """
    dc["time_step"] = dc.time - dc.time.shift(1)
    dc["speed_mps"] = dc.speed * 1000 / 3600
    dc["dist"] = (dc.time_step * dc.speed_mps) / 1000
    # uses dist as an intermediate step to calculate time_step distance traveled.
    dc["distance"] = (dc.time_step * dc.speed_mps) / 1000 + dc['dist'].shift(1).cumsum()
    dc["grade"] = np.where((dc.speed == 0) & (dc.speed.shift(1) == 0), 0, (
        (dc.elevation - dc.elevation.shift(1)) / (((dc.speed + dc.speed.shift(1)) / 2) * dc.time_step)))
    dc["slope"] = np.arctan(dc.grade)
    dc = dc.fillna(0)

    dc["f_g"] = np.where(dc.speed == 0, 0, gvw * GRAVITY * np.sin(dc.slope))
    dc["f_aero"] = (1 / 2) * RHO_AIR * frontal_area * vehicle_cd * (dc.speed_mps ** 2)
    dc["f_rr"] = gvw * GRAVITY * np.cos(dc.slope) * wheel_crr
    dc["f_accel"] = gvw * (dc.speed_mps - dc.speed_mps.shift(1)) / dc.time_step
    dc.f_accel.iat[0] = 0
    dc["f_trac"] = dc.f_g + dc.f_aero + dc.f_rr + dc.f_accel

    return dc


def dc_power(dc):
    """Calculate power from forces.

      Args:
          dc (dataframe object): Main model dataframe

      Returns:
          dc (dataframe object):  Main model DF with added calculated fields
      """
    dc["p_g"] = dc.f_g * dc.speed_mps / 1000
    dc["p_aero"] = dc.f_aero * dc.speed_mps / 1000
    dc["p_rr"] = dc.f_rr * dc.speed_mps / 1000
    dc["p_accel"] = dc.f_accel * dc.speed_mps / 1000
    dc["p_trac"] = dc.p_g + dc.p_aero + dc.p_rr + dc.p_accel

    return dc


def aux_trans_power(dc, pwr_aux_base, pwr_aux_pcnt, pwr_conv_eff, trans_eff, hvac_load):
    """Calculate the transmission power

      Args:
          dc (dataframe object): Main model dataframe
          pwr_aux_base (float): Auxiliary Power Draw Baseload (kW)
          pwr_aux_pcnt (float): Auxilliary Power Draw Variable in Decimal (0 to 1)
          pwr_conv_eff (float): Power Converter Efficiency in Decimal (0 to 1)
          trans_eff (float): Transmission efficiency in Decimal (0 to 1)
          hvac_load (float): HVAC Load (kW) [Calculated in hvac_heating()]

      Returns:
          dc (dataframe object):  Main model DF with added calculated fields
      """
    ###
    dc["hvac_load"] = hvac_load
    # AUXILIARY POWER
    dc["pwr_aux_net"] = pwr_aux_base + (pwr_aux_pcnt * dc.p_trac) + dc["hvac_load"]
    dc["pwr_aux_gross"] = dc.pwr_aux_net / pwr_conv_eff

    # ## TRANSMISSION POWER
    dc["pwr_trans_trac"] = (dc.pwr_aux_gross + dc.p_trac) / trans_eff
    dc["pwr_trans_loss"] = dc.pwr_trans_trac - (dc.p_trac + dc.pwr_aux_gross)
    dc["pwr_trans_dch"] = np.where(dc.pwr_trans_trac > 0, dc.pwr_trans_trac, 0)
    dc["pwr_trans_deccel"] = np.where(dc.pwr_trans_trac < 0, dc.pwr_trans_trac, 0)

    return dc


def engine_priorities(dc, engine_type, ic_engine, fc_engine, hybrid):
    """Assigning Drivetrain Priories. Will assign based on engine type

      Args:
          dc (dataframe object): Main df object
          engine_type (string): Engine type of vehicle
          ic_engine (list object): list of internal combustion engine types (strings)
          fc_engine (list object): list of fuel cell engine types (strings)
          hybrid (list object): list of hybrid engine types (strings)

      Returns:
          dc (dataframe object):  Main model DF with added calculated fields
      """
    # DRIVE TRAIN PRIORITIES
    dc["ic_priority"] = np.where(engine_type in ic_engine, 1, 0)
    dc["fc_priority"] = np.where(engine_type in fc_engine, 1, 0)

    # ess conditions: 2 for Hybrid, 1 for Battery-Electric, 0 for other. Confirm that FC not included in BE
    conditions = [engine_type in hybrid, engine_type == "be", (engine_type not in hybrid) & (engine_type != "be")]
    ess = [2, 1, 0]
    dc["ess_priority"] = int(np.select(conditions, ess, default=np.nan))

    return dc


def ice_power_limit(dc, ic_pwr_max, ic_time_full, ic_time_zero):
    """Calculate Internal Combustion Power and Steps

      Args:
          dc (dataframe object): Main df object
          ic_pwr_max (float): Max IC power output (kW)
          ic_time_full (float): Time to full IC power (s)
          ic_time_zero (float): Time to zero IC power (s)

      Returns:
         dc (dataframe object):  Main model DF with added calculated fields
      """
    # ## ICE POWER LIMIT
    # initialize ic_pwr_out and ic_pwr_ramp_columns
    dc["ic_pwr_out"] = 0
    dc["ic_pwr_ramp"] = 0

    # initialize rows for ic_pwr_out and ic_pwr_ramp calcs
    dc["ic_pwr_req"] = np.where(dc.ic_priority == 1, dc.pwr_trans_dch, 0)
    dc["ic_step_up"] = dc.time_step * ic_pwr_max / ic_time_full
    dc["ic_step_down"] = dc.time_step * ic_pwr_max / ic_time_zero

    # if/elif/else conditions for ic_pwr_ramp
    m1 = ((dc.ic_pwr_req - dc.ic_pwr_out.shift(1)) >= 0) & ((dc.ic_pwr_req - dc.ic_pwr_out.shift(1)) > dc.ic_step_up)
    m2 = ((dc.ic_pwr_req - dc.ic_pwr_out.shift(1)) >= 0)
    m3 = np.abs(dc.ic_pwr_req - dc.ic_pwr_out.shift(1)) > dc.ic_step_down
    ic_pwr_ramp_conditions = [m1, m2, m3]
    ic_pwr_ramp_vals = [(dc.ic_pwr_out.shift(1) + dc.ic_step_up), dc.ic_pwr_req,
                        (dc.ic_pwr_out.shift(1) - dc.ic_step_down)]

    ###if/elif/else conditions for ic_pwr_out
    m4 = (dc.ic_priority == 1) & (ic_pwr_max >= dc.ic_pwr_ramp)
    m5 = (dc.ic_priority == 1)
    ic_pwr_out_conditions = [m4, m5]
    ic_pwr_out_vals = [dc.ic_pwr_ramp, ic_pwr_max]
    dc["ic_pwr_ramp"] = np.select(ic_pwr_ramp_conditions, ic_pwr_ramp_vals, default=dc.ic_pwr_req)
    dc["ic_pwr_out"] = np.select(ic_pwr_out_conditions, ic_pwr_out_vals, default=0)
    dc.fillna(0, inplace=True)

    return dc


def emotor_requirements(dc, emotor_eff_map, emotor_pwr_max, emotor_max_eff, emotor_time_full):
    """Calculate Electric Motor Power Requirements

      Args:
          dc (dataframe object): Main df object
          emotor_eff_map (dataframe object): Datafrane containing emortor efficiency lookup values
          emotor_pwr_max (float): Max emotor power (kW)
          emotor_max_eff (float): Decimal value for max efficiency from emotor (0 to 1)
          emotor_time_full (float): Time to emotor full power (s)

      Returns:
         dc (dataframe object):  Main model DF with added calculated fields
      """
    # ## ELECTRIC MOTOR REQUIREMENTS
    dc["elec_pwr_out"] = emotor_pwr_max

    # loading efficiency map from "powerflow efficiency maps hidden tab"
    emotor_eff_map.columns = ["emotor_pwr_out_pct", ">75kw", "7.5kw"]
    emotor_eff_map["emotor_efficiency"] = emotor_eff_map[">75kw"] + (emotor_max_eff - np.max(emotor_eff_map[">75kw"]))
    dc["elec_pwr_req"] = np.where((dc.ic_priority == 1) & (dc.ess_priority > 0), dc.pwr_trans_trac - dc.ic_pwr_out,
                                  dc.pwr_trans_trac)
    dc["elec_max_step"] = dc.time_step * emotor_pwr_max / emotor_time_full

    # conditions for elec_pwr_ramp
    em_m1 = ((dc.elec_pwr_req - dc.elec_pwr_out.shift(1)) >= 0) & (
        (dc.elec_pwr_req - dc.elec_pwr_out.shift(1)) > dc.elec_max_step)
    em_m2 = (dc.elec_pwr_req - dc.elec_pwr_out.shift(1)) >= 0
    em_m3 = dc.elec_pwr_req > dc.elec_pwr_out.shift(1)
    elec_pwr_ramp_conditions = [em_m1, em_m2, em_m3]
    elec_pwr_ramp_vals = [(dc.elec_pwr_out.shift(1) + dc.elec_max_step), dc.elec_pwr_req, dc.elec_pwr_out.shift(1)]
    dc["elec_pwr_ramp"] = np.select(elec_pwr_ramp_conditions, elec_pwr_ramp_vals, default=dc.elec_pwr_req)

    ###does the intial value have to be set to 0? DOUBLE CHECK ~~~~~~~~~~~~~~~~~~~~~~~~~
    dc.elec_pwr_ramp.iat[0] = 0

    # intermediate step to compare emotor_eff_map for merge
    dc["emotor_pwr_out_pct"] = np.abs(dc.elec_pwr_ramp) / emotor_pwr_max
    # dc.fillna(0, inplace=True)
    dc = pd.merge_asof(dc.sort_values('emotor_pwr_out_pct'), emotor_eff_map, on="emotor_pwr_out_pct")
    dc = dc.drop([">75kw", "7.5kw"], axis=1)
    dc["elec_pwr_map"] = np.where(dc.elec_pwr_ramp >= 0, dc.elec_pwr_ramp / dc.emotor_efficiency,
                                  dc.elec_pwr_ramp * dc.emotor_efficiency)
    dc["elec_pwr_clip"] = dc.elec_pwr_req - dc.elec_pwr_ramp
    dc["elec_pwr_loss"] = np.abs(dc.elec_pwr_ramp) * (1 - dc.emotor_efficiency)

    # conditions for elec_pwr_out
    po_m1 = ((dc.fc_priority + dc.ess_priority) > 0) & (emotor_pwr_max >= dc.elec_pwr_map)
    po_m2 = (dc.fc_priority == 1) | (dc.ess_priority == 1)
    elec_pwr_out_conditions = [po_m1, po_m2]
    elec_pwr_out_vals = [dc.elec_pwr_map, emotor_pwr_max]
    dc["elec_pwr_out"] = np.select(elec_pwr_out_conditions, elec_pwr_out_vals, default=0)
    dc["elec_pwr_out_did"] = np.where(dc.elec_pwr_out >= 0, dc.elec_pwr_out, 0)
    dc["elec_pwr_out_chrg"] = np.where(dc.elec_pwr_out < 0, dc.elec_pwr_out, 0)
    dc = dc.sort_values(by=['time'])

    return dc


def fuel_cell_init_beb(dc, pwr_conv_eff, fc_pwr_max, fc_time_full, fc_time_zero):
    """Function to initialize values for beb fc columns for future calculations

      Args:
          dc (dataframe object): Main df object
          pwr_conv_eff (float): Power Converter Efficiency (kW)
          fc_pwr_max (float): Fuel Cell Max Engine Power (kW)
          fc_time_full (float): FC time to full power (s)
          fc_time_zero (float): FC time to zero power from full (s)
      Return:
          dc (dataframe object):  Main model DF with added calculated fields
      """
    # ## FUEL CELL POWER
    ###initialize ic_pwr_out and ic_pwr_ramp_columns
    dc["fc_pwr_out"] = 0
    dc["fc_pwr_ramp"] = 0

    ###initialize rows for fc_pwr_out and fc_pwr_ramp calcs
    dc["fc_pwr_req"] = np.where(dc.fc_priority == 1, dc.pwr_trans_dch, 0)
    dc["fc_pwr_conv_loss"] = dc.fc_pwr_req * (1 - pwr_conv_eff)
    dc["fc_step_up"] = dc.time_step * fc_pwr_max / fc_time_full
    dc["fc_step_down"] = dc.time_step * fc_pwr_max / fc_time_zero

    ###if/elif/else conditions for fc_pwr_ramp
    m1 = ((dc.fc_pwr_req - dc.fc_pwr_out.shift(1)) >= 0) & ((dc.fc_pwr_req - dc.fc_pwr_out.shift(1)) > dc.fc_step_up)
    m2 = ((dc.fc_pwr_req - dc.fc_pwr_out.shift(1)) >= 0)
    m3 = np.abs(dc.fc_pwr_req - dc.fc_pwr_out.shift(1)) > dc.fc_step_down
    fc_pwr_ramp_conditions = [m1, m2, m3]
    fc_pwr_ramp_vals = [(dc.fc_pwr_out.shift(1) + dc.fc_step_up), dc.fc_pwr_req,
                        (dc.fc_pwr_out.shift(1) - dc.fc_step_down)]

    ###if/elif/else conditions for fc_pwr_out
    m4 = (dc.fc_priority == 1) & (fc_pwr_max >= dc.fc_pwr_ramp)
    m5 = (dc.fc_priority == 1)
    fc_pwr_out_conditions = [m4, m5]
    fc_pwr_out_vals = [dc.fc_pwr_ramp, fc_pwr_max]

    dc["fc_pwr_ramp"] = np.select(fc_pwr_ramp_conditions, fc_pwr_ramp_vals, default=dc.ic_pwr_req)
    dc["fc_pwr_out"] = np.select(fc_pwr_out_conditions, fc_pwr_out_vals, default=0)
    dc["fc_pwr_clip"] = np.where(dc.fc_priority == 1, dc.fc_pwr_req - dc.fc_pwr_out, 0)

    ### Review the fc_pwr_req == 0 condition here. It is not in the excel model, just an assumption based on model ~~~~~~~~~~~~~~~~~~~
    # dc["fc_pwr_deficit"] = np.where( dc.fc_pwr_req == 0, 0, dc.elec_pwr_out - dc.fc_pwr_out)
    dc["fc_pwr_deficit"] = dc.elec_pwr_out - dc.fc_pwr_out
    dc.fillna(0, inplace=True)

    return dc


def fuel_cell_requirements_beb(dc, pwr_conv_eff, fc_pwr_max, fc_time_full, fc_time_zero):
    # ## FUEL CELL POWER
    ###initialize ic_pwr_out and ic_pwr_ramp_columns
    dc["fc_pwr_out"] = 0
    dc["fc_pwr_ramp"] = 0

    ###initialize rows for fc_pwr_out and fc_pwr_ramp calcs
    dc["fc_pwr_req"] = np.where(dc.fc_priority == 1, dc.pwr_trans_dch, 0)
    fc_m1 = (dc.fc_priority == 1) & (dc.ess_priority == 0)
    fc_m2 = (dc.fc_priority == 1) & ((dc.elec_pwr_out_did - dc.engine_adj_kw) > 0)
    fc_pwr_req_conditions = [fc_m1, fc_m2]

    fc_v1 = dc.elec_pwr_out_did / pwr_conv_eff
    fc_v2 = (dc.elec_pwr_out_did - dc.engine_adj_kw) / pwr_conv_eff
    fc_pwr_req_vals = [fc_v1, fc_v2]
    dc["fc_pwr_req"] = np.select(fc_pwr_req_conditions, fc_pwr_req_vals, default=0)

    dc["fc_pwr_conv_loss"] = dc.fc_pwr_req * (1 - pwr_conv_eff)
    dc["fc_step_up"] = dc.time_step * fc_pwr_max / fc_time_full
    dc["fc_step_down"] = dc.time_step * fc_pwr_max / fc_time_zero

    ###if/elif/else conditions for fc_pwr_ramp
    m1 = ((dc.fc_pwr_req - dc.fc_pwr_out.shift(1)) >= 0) & ((dc.fc_pwr_req - dc.fc_pwr_out.shift(1)) > dc.fc_step_up)
    m2 = ((dc.fc_pwr_req - dc.fc_pwr_out.shift(1)) >= 0)
    m3 = np.abs(dc.fc_pwr_req - dc.fc_pwr_out.shift(1)) > dc.fc_step_down
    fc_pwr_ramp_conditions = [m1, m2, m3]
    fc_pwr_ramp_vals = [(dc.fc_pwr_out.shift(1) + dc.fc_step_up), dc.fc_pwr_req,
                        (dc.fc_pwr_out.shift(1) - dc.fc_step_down)]

    dc["fc_pwr_ramp"] = np.select(fc_pwr_ramp_conditions, fc_pwr_ramp_vals, default=dc.ic_pwr_req)

    ###if/elif/else conditions for fc_pwr_out
    m4 = (dc.fc_priority == 1) & (fc_pwr_max >= dc.fc_pwr_ramp)
    m5 = (dc.fc_priority == 1)
    fc_pwr_out_conditions = [m4, m5]
    fc_pwr_out_vals = [dc.fc_pwr_ramp, fc_pwr_max]

    dc["fc_pwr_out"] = np.select(fc_pwr_out_conditions, fc_pwr_out_vals, default=0)
    dc["fc_pwr_clip"] = np.where(dc.fc_priority == 1, dc.fc_pwr_req - dc.fc_pwr_out, 0)

    ### Review the fc_pwr_req == 0 condition here. It is not in the excel model, just an assumption based on model ~~~~~~~~~~~~~~~~~~~
    # dc["fc_pwr_deficit"] = np.where( dc.fc_pwr_req == 0, 0, dc.elec_pwr_out - dc.fc_pwr_out)
    dc["fc_pwr_deficit"] = dc.elec_pwr_out - dc.fc_pwr_out
    dc.fillna(0, inplace=True)

    return dc


def regen_map_calculations_beb(dc, regen_map, regen_eff_max, regen_a, regen_b):
    """Calculate the regen power

      Args:
          dc (dataframe object): Main df object
          regen_map (dataframe object): Lookup values to calculate regeneration percentage
          regen_eff_max (float): Regen braking efficiency in Decimal (0 to 1)
          regen_a (float): Constant used in regen percent calculation
          regen_b (float): Constant used in regen percent calculation

      Returns:
          dc (dataframe object):  Main model DF with added calculated fields
      """
    ###regen curve calculation
    regen_map["regen_pcnt"] = regen_eff_max / (1 + regen_a * np.exp((-1 * regen_b) * (regen_map.speed_mph + 1)))
    dc = pd.merge_asof(dc.sort_values('speed_mps'), regen_map, on="speed_mps")
    dc["ess_pwr_regen_map"] = dc.regen_pcnt * dc.ess_pwr_chrg_net

    dc = dc.sort_values(by=['time'])
    dc = dc.reset_index(drop=True)
    dc = dc.drop(["speed_mph", "speed_kph"], axis=1)

    return dc


def ess_state_of_charge_beb(dc, ess_init_soc, ess_cap, ess_min_soc, ess_pwr_max, max_regen_pwr, ess_round_trip_eff,
                            ess_target_soc, ess_soc_gain):
    """Calculate the State of Charge of the Battery

      Args:
          dc (dataframe object): Main df object
          ess_init_soc (float): Inital state of charge in percent (%)
          ess_cap (float): Energy Storage Capacity (kWh)
          ess_min_soc (float): Minimum battery state of charge in percent (%)
          ess_pwr_max (float): Energy Storage Power (kW)
          max_regen_pwr (float): Max regen braking power (kW)
          ess_round_trip_eff (float): Energy Storage Round Trip Efficiency in decimal (0 to 1)
          ess_target_soc (float): Hybrid Battery SOC target (kW)
          ess_soc_gain (float): SOC control proportional gain decimal (0 to 1)

      Returns:
         dc (dataframe object):  Main model DF with added calculated fields
      """
    # initialize columns
    dc["ess_soc"] = ess_init_soc
    dc["ess_cur_cap"] = ess_init_soc * ess_cap / 100
    dc["ess_soc_lim"] = np.where(dc.ess_cur_cap.shift(1) / ess_cap <= ess_min_soc, 0, dc.ess_pwr_dis_gross)
    dc["ess_pwr_dis_out"] = np.where(ess_pwr_max >= dc.ess_soc_lim, dc.ess_soc_lim, ess_pwr_max)
    ##issue is a results of ess_soc not decending
    dc["ess_pwr_soc_lim"] = np.where(dc.ess_pwr_regen_map < 0, dc.ess_pwr_regen_map, 0)
    # dc["ess_pwr_soc_lim"] =  dc.ess_pwr_regen_map
    dc["ess_pwr_chrg_out"] = np.where(((dc.ess_priority > 0) & (max_regen_pwr >= np.abs(dc.ess_pwr_soc_lim))),
                                      dc.ess_pwr_soc_lim, 0)
    dc["ess_pwr_dis_clip"] = dc.ess_pwr_dis_gross - dc.ess_pwr_dis_out
    dc["mech_brk_pwr"] = dc.pwr_trans_deccel - dc.ess_pwr_chrg_out
    dc["ess_pwr_chrg_clip"] = dc.ess_pwr_chrg_net - dc.ess_pwr_chrg_out
    dc["ess_pwr_chrg_loss"] = np.abs(dc.ess_pwr_chrg_out) * (1 - np.sqrt(ess_round_trip_eff))
    dc["ess_cur_cap_dis"] = (dc.ess_pwr_chrg_out + dc.ess_pwr_dis_out) * dc.time_step / 3600
    ###make rules around intial value, if ess_lim_soc == 0
    ecc_m1 = dc.time == 0
    ecc_m2 = (dc.ess_soc_lim == 0) & (dc.ess_pwr_chrg_out == 0)
    ecc_conditions = [ecc_m1, ecc_m2, ]
    ecc_v1 = ess_init_soc
    ecc_v2 = ess_cap * ess_min_soc / 100
    ecc_values = [ecc_v1, ecc_v2, ]
    dc["ess_cur_cap"] = np.where((np.select(ecc_conditions, ecc_values, default=(
        dc.ess_cur_cap.shift(1) - dc.ess_cur_cap_dis.cumsum()))) <= ess_cap * ess_min_soc, ess_cap * ess_min_soc, (
                                     np.select(ecc_conditions, ecc_values,
                                               default=(dc.ess_cur_cap.shift(1) - dc.ess_cur_cap_dis.cumsum()))))
    # dc["ess_cur_cap"] = np.select( ecc_conditions, ecc_values, default= (dc.ess_cur_cap.shift(1) - dc.ess_cur_cap_dis.cumsum()) )
    dc["ess_soc_lim"] = np.where(dc.ess_cur_cap.shift(1) / ess_cap <= ess_min_soc, 0, dc.ess_pwr_dis_gross)
    dc["ess_pwr_dis_out"] = np.where(ess_pwr_max >= dc.ess_soc_lim, dc.ess_soc_lim, ess_pwr_max)
    dc["ess_soc"] = dc.ess_cur_cap / ess_cap
    # Error in SOC vs Target
    dc["soc_error"] = dc.ess_soc - ess_target_soc
    # engine_adj_kw conditions ~~~~in the model the SOC error is not percent, scale by 100 ~~~~~~~~~~~~~~~~~~~~VERIFY THAT THIS LOGIC IS CORRECT ~~~~~~~~~~~~~~~~~
    eak_m1 = dc.soc_error * ess_pwr_max * ess_soc_gain >= max_regen_pwr
    eak_m2 = dc.soc_error * max_regen_pwr * ess_soc_gain <= (-1) * max_regen_pwr
    engine_adj_kw_conditions = [eak_m1, eak_m2]
    # engine_adj_kw values
    eak_v1 = max_regen_pwr
    eak_v2 = (-1) * max_regen_pwr
    eak_default = dc.soc_error * max_regen_pwr * ess_soc_gain
    engine_adj_kw_vals = [eak_v1, eak_v2]
    # Engine Adjust Signal to Balance SOC
    dc["engine_adj_kw"] = np.select(engine_adj_kw_conditions, engine_adj_kw_vals, default=eak_default)

    return dc


def fuel_consumption(dc, ic_pwr_max, fc_pwr_max, h2_lhv):
    """Fuel Consumption Calculation

      Args:
          dc (dataframe object): Main df object
          ic_pwr_max (_type_): Internal Combustion Engine Power (kW)
          fc_pwr_max (_type_): Fuel Cell Engine Power (kW)
          h2_lhv (float): Hydrogen Heating Value LHV (kWh/kg)

      Returns:
          dc (dataframe object):  Main model DF with added calculated fields
      """
    # engine_eff_map_lookup conditions
    eeml_m1 = dc.ic_priority == 1
    eeml_m2 = dc.fc_priority == 1
    engine_eff_map_lookup_conditions = [eeml_m1, eeml_m2]

    # engine_eff_map_lookup values
    eeml_v1 = dc.ic_pwr_out / ic_pwr_max
    eeml_v2 = dc.fc_pwr_out / fc_pwr_max
    engine_eff_map_lookup_vals = [eeml_v1, eeml_v2]

    dc["engine_eff_map_lookup"] = np.select(engine_eff_map_lookup_conditions, engine_eff_map_lookup_vals, default=0)

    ### Create Lookup Dataframe
    power_out = np.array([0, 0.005, 0.015, 0.04, 0.06, 0.10, 0.14, 0.20, 0.40, 0.60, 0.80, 1.00])
    si_table = np.array([0.1, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.30])
    atkinson_table = np.array([0.1, 0.12, 0.28, 0.35, 0.38, 0.39, 0.40, 0.40, 0.38, 0.37, 0.36, 0.35])
    diesel_table = np.array([0.1, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34])
    cng_table = np.array([0.1, 0.13, 0.19, 0.25, 0.30, 0.37, 0.39, 0.40, 0.39, 0.36, 0.34, 0.32])
    sofc_table = np.array([0.1, 0.11, 0.14, 0.20, 0.25, 0.35, 0.41, 0.49, 0.55, 0.61, 0.60, 0.55])
    pem_fc_table = np.array([0.1, 0.30, 0.36, 0.45, 0.50, 0.56, 0.58, 0.60, 0.58, 0.57, 0.55, 0.54])

    engine_eff_dict = {
        'power_out_%': power_out,
        'si': si_table,
        'atkinson': atkinson_table,
        'diesel': diesel_table,
        'cng': cng_table,
        'sofc': sofc_table,
        'pem_fuel_cell': pem_fc_table,
    }

    engine_eff_map = pd.DataFrame(engine_eff_dict)

    dc = dc.sort_values('engine_eff_map_lookup', ascending=True)

    dc = pd.merge_asof(left=dc, right=engine_eff_map, left_on="engine_eff_map_lookup", right_on='power_out_%',
                       suffixes=('', '_eff'), direction='nearest')

    dc = dc.sort_values('time')

    #### Replace with model input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    engine_select = 'sofc'

    if engine_select == 'si':
        dc['engine_efficiency'] = dc['si']
    elif engine_select == 'atkinson':
        dc['engine_efficiency'] = dc['atkinson']
    elif engine_select == 'diesel':
        dc['engine_efficiency'] = dc['diesel']
    elif engine_select == 'cng':
        dc['engine_efficiency'] = dc['cng']
    elif engine_select == 'sofc':
        dc['engine_efficiency'] = dc['sofc']
    elif engine_select == 'pem_fc':
        dc['engine_efficiency'] = dc['pem_fuel_cell']
    else:
        dc['engine_efficiency'] = None

    # ic_energy_net conditions
    ien_m1 = dc.time == 0
    ien_m2 = dc.ic_priority == 1
    ic_energy_net_conditions = [ien_m1, ien_m2]

    # ic_energy_net values
    ien_v1 = 0
    ien_v2 = (dc.ic_pwr_out * dc.time_step / 3600) / (dc.engine_efficiency)
    ic_energy_net_vals = [ien_v1, ien_v2]

    dc["ic_energy_net"] = np.select(ic_energy_net_conditions, ic_energy_net_vals, default=0)

    # fc_energy_net conditions
    fen_m1 = dc.time == 0
    fen_m2 = dc.fc_priority == 1
    fc_energy_net_conditions = [fen_m1, fen_m2]

    # fc_energy_net values
    fen_v1 = 0
    fen_v2 = (dc.fc_pwr_out * dc.time_step / 3600) / (dc.engine_efficiency)
    fc_energy_net_vals = [fen_v1, fen_v2]

    dc["fc_energy_net"] = np.select(fc_energy_net_conditions, fc_energy_net_vals, default=0)

    # ess_energy_net conditions
    een_m1 = dc.time == 0
    een_m2 = dc.ess_priority == 1
    ess_energy_net_conditions = [een_m1, een_m2]

    # ess_energy_net values
    een_v1 = 0
    een_v2 = ((dc.ess_pwr_dis_out + dc.ess_pwr_chrg_out) * dc.time_step / 3600)
    ess_energy_net_vals = [een_v1, een_v2]

    dc["ess_energy_net"] = np.select(ess_energy_net_conditions, ess_energy_net_vals, default=0)

    # dc["cumulative_fc_energy"] = np.where(dc.time == 0, 0, dc.fc_energy_net + dc.cumulative_fc_energy.shift(1))
    dc["cumulative_fc_energy"] = dc.fc_energy_net.cumsum()
    dc["h2_consumption"] = dc.cumulative_fc_energy / h2_lhv

    return dc


def results_table(dc, ess_round_trip_eff, pwr_conv_eff, trans_eff):
    """Create Dataframe of results

      Args:
          dc (dataframe object): Main df object
          ess_round_trip_eff (float): Energy Storage Round Trip Efficiency (0 to 1)
          pwr_conv_eff (float): Power Converter Efficiency (0 to 1)
          trans_eff (float): Transmission efficiency (0 to 1)

      Returns:
          results (dataframe object):  Results DF with results values.
      """

    # # MODEL RESULTS
    # ### Traction Energy and IC Energy Results Calculations
    ### Traction Energy Results
    propulsion = dc.pwr_trans_dch * dc.time_step / 3600
    decceleration = dc.pwr_trans_deccel * dc.time_step / 3600
    transmission_losses = dc.pwr_trans_loss * dc.time_step / 3600

    ### IC Energy Results
    gross_ic = dc.ic_energy_net
    net_ic = dc.ic_pwr_req * dc.time_step / 3600
    ic_efficiency_loss = gross_ic - net_ic
    ic_ramp_clipped_energy = (dc.ic_pwr_req - dc.ic_pwr_out) * dc.time_step / 3600

    results = pd.DataFrame(
        data={"Propulsion": propulsion, "Decceleration": decceleration, "Transmission Losses": transmission_losses,
              "Gross ICE": gross_ic, "Net ICE": net_ic, "ICE Effiency Loss": ic_efficiency_loss,
              "ICE Ramp Clipped Energy": ic_ramp_clipped_energy})

    # ### FC Energy Results Calculations
    ### FC Energy Results
    gross_fc_energy = dc.fc_energy_net
    net_fc_energy = dc.fc_pwr_req * dc.time_step / 3600
    fc_efficiency_loss = gross_fc_energy - net_fc_energy
    fc_ramp_clipped_energy = (dc.fc_pwr_req - dc.fc_pwr_out) * dc.time_step / 3600
    fc_power_converter_loss = dc.fc_pwr_conv_loss * dc.time_step / 3600
    results["Gross FC Energy"] = gross_fc_energy
    results["Net FC Energy"] = net_fc_energy
    results["FC Efficiency Loss"] = fc_efficiency_loss
    results["FC Ramp Clipped Energy"] = fc_ramp_clipped_energy
    results["FC Power Converter Loss"] = fc_power_converter_loss

    ## Electric Motor Energy
    results["Gross E-Motor Propulsion Energy"] = dc.elec_pwr_out_did * dc.time_step / 3600
    results["Net E-Motor Propulsion Energy"] = dc.emotor_efficiency * dc.elec_pwr_out_did * dc.time_step / 3600
    results["E-Motor Propulsion Efficiency Loss"] = results["Gross E-Motor Propulsion Energy"] - results[
        "Net E-Motor Propulsion Energy"]
    results["E-Motor Clipped Energy"] = dc.elec_pwr_clip * dc.emotor_efficiency * dc.time_step / 3600
    results["Gross E-Motor Regen Energy"] = dc.elec_pwr_out_chrg / dc.emotor_efficiency * dc.time_step / 3600
    results["Net E-Motor Regen Energy"] = dc.elec_pwr_out_chrg * dc.time_step / 3600
    results["E-Motor Regen Efficiency Loss"] = (
                                                   dc.elec_pwr_out_chrg / dc.emotor_efficiency - dc.elec_pwr_out_chrg) * dc.time_step / 3600

    # ### ESS Discharge Energy
    results["Gross ESS Discharge Energy"] = (dc.ess_pwr_dis_out * dc.time_step / 3600) / np.sqrt(ess_round_trip_eff)
    results["Net ESS Discharge Energy"] = (dc.ess_pwr_dis_out * dc.time_step / 3600)
    results["ESS Discharge Efficiency Loss"] = (dc.ess_pwr_dis_out / np.sqrt(
        ess_round_trip_eff) - dc.ess_pwr_dis_out) * dc.time_step / 3600  # Gross - Net
    results["ESS Discharge Clipped Energy"] = (
                                                  dc.ess_pwr_dis_req - dc.ess_pwr_dis_out) * pwr_conv_eff * dc.emotor_efficiency * dc.time_step / 3600
    results["ESS Power Converter Discharge Loss"] = dc.ess_pwr_conv_dis_loss * dc.time_step / 3600

    ### ESS Charge Energy
    results["Gross ESS Charge Energy"] = ((dc.ess_pwr_chrg_out * (pwr_conv_eff ** 2) * dc.emotor_efficiency) * (
        dc.time_step / 3600)) / np.sqrt(ess_round_trip_eff)
    results["Net ESS Charge Energy"] = (dc.ess_pwr_chrg_out * (pwr_conv_eff ** 2) * dc.emotor_efficiency) * (
        dc.time_step / 3600)
    results["ESS Charge Efficiency Loss"] = results["Gross ESS Charge Energy"] - results["Net ESS Charge Energy"]
    results["ESS Charge Clipped Energy"] = (dc.ess_pwr_chrg_req - dc.ess_pwr_chrg_out) * dc.time_step / 3600
    results["ESS Power Converter Charge Loss"] = dc.ess_pwr_conv_chrg_loss * dc.time_step / 3600

    # ess_charge_energy_results = results[["Gross ESS Charge Energy", "Net ESS Charge Energy", "ESS Charge Efficiency Loss", "ESS Charge Clipped Energy", "ESS Power Converter Charge Loss" ]]
    # ###ESS Clipped Energy value is due to what happens after soc runs out. check on this
    # ess_charge_energy_results.sum()

    ### Auxilliary Loads Energy
    results["Gross Aux Energy"] = dc.pwr_aux_gross * dc.time_step / 3600
    results["Net Aux Energy"] = dc.pwr_aux_net * dc.time_step / 3600
    results["Aux Load Efficiency Loss"] = (dc.pwr_aux_gross - dc.pwr_aux_net) * dc.time_step / 3600

    ### Transmission Energy
    results["Gross Transmission Energy"] = dc.pwr_trans_dch * dc.time_step / 3600
    results["Net Transmission Energy"] = dc.pwr_trans_dch * dc.time_step / 3600 * trans_eff
    results["Transmission Efficiency Loss"] = results["Gross Transmission Energy"] - results["Net Transmission Energy"]
    results["Mechanical Braking Energy"] = (dc.pwr_trans_deccel - dc.ess_pwr_chrg_out * (
        pwr_conv_eff ** 2) * dc.emotor_efficiency) * dc.time_step / 3600

    return results


def dashboard_graphs(dc, ess_min_soc, engine_type, FUEL_TANK_SIZE):
    """Create graphs for results page

      Args:
          dc (_type_): _description_
          ess_min_soc (float): Battery Minimum State of Charge (%)
          engine_type (string): Engine type of vehicle
          FUEL_TANK_SIZE (float): Size of H2 storage tank

      Returns:
          dc (dataframe object):  Main model DF with added calculated fields
      """
    ###Bokeh elev vs time graph###
    curdoc().theme = 'dark_minimal'
    speed_elev_time = figure(width=675, height=300)
    hours = dc.time / 3600
    speed_elev_time.line(hours, dc.speed, legend_label='Cycle Speed (mps)', color="#154A8B", line_width=1)
    speed_elev_time.line(hours, dc.elevation, legend_label='Cycle Elevation (m)', line_width=2, color="red")
    speed_elev_time.xaxis.axis_label = "Time (hr)"
    speed_elev_time.yaxis.axis_label = "Speed (mps)"
    speed_elev_time.add_layout(LinearAxis(axis_label="Elevation (m)"), 'right')

    dc_script, dc_div = components(speed_elev_time)
    if engine_type == 'be':
        ess_cap = figure(width=675, height=300)
        ess_cap.line(hours, dc.pwr_trans_trac, legend_label='Total Traction Power (kW)', line_width=1.5,
                     color="#154A8B")
        ess_cap.line(hours, dc.ess_soc, legend_label='ESS State of Charge (%)', line_width=2, y_range_name="foo",
                     color="red")
        ess_cap.line(hours, ess_min_soc, legend_label='Minimum State of Charge (%)', color="black", line_dash="dashed",
                     line_width=2, y_range_name="foo")
        ess_cap.xaxis.axis_label = "Time (hr)"
        ess_cap.yaxis.axis_label = "Total Traction Power (kW)"
        ess_cap.extra_y_ranges = {"foo": Range1d(start=-20, end=100)}
        ess_cap.add_layout(LinearAxis(y_range_name="foo", axis_label="ESS State of Charge (%)"), 'right')

        ess_script, ess_div = components(ess_cap)
    else:
        ess_cap = figure(width=775, height=300)
        ess_cap.line(hours, dc.pwr_trans_trac, legend_label='Total Traction Power (kW)', line_width=1.5,
                     color="#154A8B")
        ess_cap.line(hours, dc.h2_consumption, legend_label='H2 Consumption (kg-H2)', line_width=2, y_range_name="h2",
                     color="red")
        ess_cap.line(hours, FUEL_TANK_SIZE, legend_label='H2 Tank Capacity', color="black", line_dash="dashed",
                     line_width=2, y_range_name="h2")
        ess_cap.xaxis.axis_label = "Time (hr)"
        ess_cap.yaxis.axis_label = "Total Traction Power (kW)"
        h2_y_height = FUEL_TANK_SIZE + 15
        pwr_height = dc.pwr_trans_trac.max()
        ess_cap.y_range = Range1d(0, pwr_height)
        ess_cap.extra_y_ranges = {"h2": Range1d(start=0, end=h2_y_height)}
        ess_cap.add_layout(LinearAxis(y_range_name="h2", axis_label='H2 Consumption (kg-H2)'), 'right')
        ess_cap.add_layout(ess_cap.legend[0], 'right')

        ess_script, ess_div = components(ess_cap)

    return dc_script, dc_div, ess_script, ess_div


def sum_stats_and_format(dc, results):
    """Create stats dataframe

      Args:
           dc (dataframe object):  Main model DF
          results (dataframe object): Dataframe of results from model calculations

      Returns:
          dc (dataframe object):  Main model DF with added calculated fields
      """
    results_stats = results.describe()
    st = results.sum()
    st.loc['Cycle'] = dc.cycle[0]
    sum_table = pd.DataFrame(data=st)
    sum_table.rename(columns={0: 'Model Results'})

    return dc, results_stats, sum_table


def fceb_fc_requirements(dc, ess_pwr_max, ess_init_soc, ess_cap, fc_pwr_max, pwr_conv_eff, fc_time_full, fc_time_zero,
                         ess_round_trip_eff, ess_time_full, regen_eff_max, regen_a, regen_b, max_regen_pwr,
                         ess_soc_gain, ess_target_soc, ess_min_soc, ess_max_soc):
    """Calculation of Fuel Cell Power Requirements and ESS State of Charge for Fuel Cell Vehicles

      Args:
          dc (_type_): _description_
          ess_pwr_max (float): Energy Storage Power (kW)
          ess_init_soc (float): Inital state of charge in percent (%)
          ess_cap (float): Energy Storage Capacity (kWh)
          fc_pwr_max (float): Fuel Cell Engine Power (kW)
          pwr_conv_eff (float): Power Converter Efficiency (0 to 1)
          fc_time_full (float): FC time to full power (s)
          fc_time_zero (float): FC time to zero power from full (s)
          ess_round_trip_eff (float): Energy Storage Round Trip Efficiency in decimal (0 to 1)
          ess_time_full (float): Energy Storage Time to full power (s)
          regen_eff_max (float): Regen braking efficiency in Decimal (0 to 1)
          regen_a (float): Constant used in regen percent calculation
          regen_b (float): Constant used in regen percent calculation
          max_regen_pwr (float): Max regen braking power (kW)
          ess_soc_gain (float): SOC control proportional gain decimal (0 to 1)
          ess_target_soc (float): Hybrid Battery SOC target (kW)
          ess_min_soc (float): Minimum battery state of charge in percent (%)
          ess_max_soc (_type_): Maximum battery state of charge in percent (%)

      Returns:
          dc (dataframe object):  Main model DF with added calculated fields
      """
    from math import sqrt
    KPH_TO_MPH = 0.621371
    dc = dc.reset_index(drop=True)

    dc["fc_pwr_req"] = 0
    dc["fc_pwr_conv_loss"] = 0
    dc["fc_step_up"] = 0
    dc["fc_step_down"] = 0
    dc["fc_pwr_ramp"] = 0
    dc["fc_pwr_out"] = 0
    dc["fc_pwr_clip"] = 0
    dc["fc_pwr_deficit"] = 0

    dc["ess_pwr_dis_req"] = 0
    dc["ess_pwr_conv_dis_loss"] = 0
    dc["ess_max_dis_step"] = 0
    dc["ess_pwr_dis_ramp"] = 0
    dc["ess_pwr_dis_gross"] = 0
    dc["ess_pwr_dis_loss"] = 0
    dc["ess_pwr_dis_clip"] = 0
    dc["ess_soc_lim"] = 0
    dc["ess_pwr_dis_out"] = ess_pwr_max

    dc["ess_pwr_chrg_req"] = 0
    dc["ess_pwr_conv_chrg_loss"] = 0
    dc["ess_pwr_chrg_net"] = 0
    dc["ess_pwr_regen_map"] = 0
    dc["ess_pwr_soc_lim"] = 0
    dc["ess_pwr_chrg_out"] = 0
    dc["ess_pwr_chrg_loss"] = 0
    dc["ess_pwr_chrg_clip"] = 0
    dc["mech_brk_pwr"] = 0

    dc["ess_soc"] = ess_init_soc
    dc["ess_cur_cap"] = ess_cap * ess_init_soc / 100
    dc["soc_error"] = 0
    dc["engine_adj_kw"] = 0

    # dc["ess_cur_cap_dis"] =  0

    dc["regen_pcnt"] = 0.005247

    for i in range(len(dc)):

        if i == 0:
            dc.loc[i, "fc_pwr_conv_loss"] = dc.fc_pwr_req[i] * (1 - pwr_conv_eff)
            dc.loc[i, "fc_step_up"] = dc.time_step[i] * fc_pwr_max / fc_time_full
            dc.loc[i, "fc_step_down"] = dc.time_step[i] * fc_pwr_max / fc_time_zero

            m4 = (dc.fc_priority[i] == 1) & (fc_pwr_max >= dc.fc_pwr_ramp[i])
            m5 = (dc.fc_priority[i] == 1)
            fc_pwr_out_conditions = [m4, m5]
            fc_pwr_out_vals = [dc.fc_pwr_ramp[i], fc_pwr_max]

            dc.loc[i, "fc_pwr_out"] = np.select(fc_pwr_out_conditions, fc_pwr_out_vals, default=0)
            dc.loc[i, "fc_pwr_clip"] = np.where(dc.fc_priority[i] == 1, dc.fc_pwr_req[i] - dc.fc_pwr_out[i], 0)
            dc.loc[i, "fc_pwr_deficit"] = dc.elec_pwr_out[i] - dc.fc_pwr_out[i]

            epcr_m1 = (dc.ic_priority[i] == 1) & (dc.ess_priority[i] > 0)
            epcr_m2 = ((dc.fc_priority[i] == 1) & (dc.ess_priority[i] > 0)) & (dc.fc_pwr_deficit[i] <= 0)
            epcr_m3 = (dc.ess_priority[i] == 1)
            ess_pwr_chrg_req_conditions = [epcr_m1, epcr_m2, epcr_m3]

            # ess_pwr_chrg_req values
            epcr_v1 = dc.elec_pwr_out_chrg[i] / pwr_conv_eff
            epcr_v2 = dc.fc_pwr_deficit[i] / pwr_conv_eff
            epcr_v3 = dc.elec_pwr_out_chrg[i] / pwr_conv_eff
            ess_pwr_chrg_req_vals = [epcr_v1, epcr_v2, epcr_v3]

            dc.loc[i, "ess_pwr_chrg_req"] = np.select(ess_pwr_chrg_req_conditions, ess_pwr_chrg_req_vals, default=0)
            dc.loc[i, "ess_pwr_chrg_net"] = dc.ess_pwr_chrg_req[i] / (np.sqrt(ess_round_trip_eff))
            dc.loc[i, "ess_pwr_conv_chrg_loss"] = dc.ess_pwr_chrg_req[i] * (1 - pwr_conv_eff)

            ess_m1 = (dc.ic_priority[i] == 1) & (dc.ess_priority[i] > 0)
            ess_m2 = ((dc.fc_priority[i] == 1) & (dc.ess_priority[i] > 0)) & (
                (dc.elec_pwr_out_did[i] - dc.fc_pwr_out[i]) <= 0)
            ess_m3 = ((dc.fc_priority[i] == 1) & (dc.ess_priority[i] > 0))
            ess_m4 = (dc.ess_priority[i] == 1)
            ess_pwr_dis_req_conditions = [ess_m1, ess_m2, ess_m3, ess_m4]

            # ess_pwr_dis_req values
            ess_v1 = (dc.elec_pwr_out_did[i] / pwr_conv_eff)
            ess_v2 = 0
            ess_v3 = ((dc.elec_pwr_out_did[i] - dc.fc_pwr_out[i]) / pwr_conv_eff)
            ess_v4 = (dc.elec_pwr_out_did[i] / pwr_conv_eff)

            ess_pwr_dis_req_vals = [ess_v1, ess_v2, ess_v3, ess_v4]

            dc.loc[i, "ess_pwr_dis_req"] = np.select(ess_pwr_dis_req_conditions, ess_pwr_dis_req_vals, default=0)
            dc.loc[i, "ess_pwr_conv_dis_loss"] = dc.ess_pwr_dis_req[i] * (1 - pwr_conv_eff)
            dc.loc[i, "ess_max_dis_step"] = dc.time_step[i] * ess_pwr_max / ess_time_full

            dc.loc[i, "ess_pwr_dis_gross"] = dc.ess_pwr_dis_ramp[i] / np.sqrt(ess_round_trip_eff)
            dc.loc[i, "ess_pwr_dis_loss"] = np.abs(dc.ess_pwr_dis_gross[i]) * (1 - np.sqrt(ess_round_trip_eff))

            dc.loc[i, "regen_pcnt"] = regen_eff_max / (
                1 + regen_a * np.exp((-1 * regen_b) * (dc.speed[i] * KPH_TO_MPH + 1)))

            dc.loc[i, "ess_pwr_regen_map"] = dc.regen_pcnt[i] * dc.ess_pwr_chrg_net[i]

            dc.loc[i, "ess_pwr_dis_out"] = np.where(ess_pwr_max >= dc.ess_soc_lim[i], dc.ess_soc_lim[i], ess_pwr_max)

            dc.loc[i, "ess_pwr_soc_lim"] = np.where(dc.ess_pwr_regen_map[i] < 0, dc.ess_pwr_regen_map[i], 0)
            # dc["ess_pwr_soc_lim"] =  dc.ess_pwr_regen_map
            dc.loc[i, "ess_pwr_chrg_out"] = np.where(
                ((dc.ess_priority[i] > 0) & (max_regen_pwr >= np.abs(dc.ess_pwr_soc_lim[i]))), dc.ess_pwr_soc_lim[i], 0)

            dc.loc[i, "ess_pwr_dis_clip"] = dc.ess_pwr_dis_gross[i] - dc.ess_pwr_dis_out[i]

            dc.loc[i, "mech_brk_pwr"] = dc.pwr_trans_deccel[i] - dc.ess_pwr_chrg_out[i]
            dc.loc[i, "ess_pwr_chrg_clip"] = dc.ess_pwr_chrg_net[i] - dc.ess_pwr_chrg_out[i]
            dc.loc[i, "ess_pwr_chrg_loss"] = abs(dc.ess_pwr_chrg_out[i]) * (1 - sqrt(ess_round_trip_eff))

            # dc.loc[ i, "ess_cur_cap_dis"] =  (dc.ess_pwr_chrg_out[i] + dc.ess_pwr_dis_out[i] ) * dc.time_step[i] / 3600

            dc.loc[i, "ess_soc_lim"] = np.where(dc.ess_cur_cap[i] / ess_cap <= ess_min_soc, 0, dc.ess_pwr_dis_gross[i])
            dc.loc[i, "ess_pwr_dis_out"] = np.where(ess_pwr_max >= dc.ess_soc_lim[i], dc.ess_soc_lim[i], ess_pwr_max)

            dc.loc[i, "ess_soc"] = ess_init_soc

            dc.loc[i, "ess_cur_cap"] = ess_cap * ess_init_soc / 100

            # Error in SOC vs Target
            dc.loc[i, "soc_error"] = dc.ess_soc[i] - ess_target_soc

            # engine_adj_kw conditions ~~~~in the model the SOC error is not percent, scale by 100 ~~~~~~~~~~~~~~~~~~~~VERIFY THAT THIS LOGIC IS CORRECT ~~~~~~~~~~~~~~~~~
            eak_m1 = dc.soc_error[i] * ess_pwr_max * ess_soc_gain >= max_regen_pwr
            eak_m2 = dc.soc_error[i] * max_regen_pwr * ess_soc_gain <= ((-1) * max_regen_pwr)
            engine_adj_kw_conditions = [eak_m1, eak_m2]

            # engine_adj_kw values
            eak_v1 = max_regen_pwr
            eak_v2 = (-1) * max_regen_pwr
            eak_default = dc.soc_error[i] * max_regen_pwr * ess_soc_gain
            engine_adj_kw_vals = [eak_v1, eak_v2]

            # Engine Adjust Signal to Balance SOC
            # dc.loc[ i, "engine_adj_kw"] = np.select( engine_adj_kw_conditions, engine_adj_kw_vals, default = eak_default )
            if dc.soc_error[i] * ess_pwr_max * ess_soc_gain >= max_regen_pwr:
                engine_adj = max_regen_pwr
            elif dc.soc_error[i] * max_regen_pwr * ess_soc_gain <= ((-1) * max_regen_pwr):
                engine_adj = ((-1) * max_regen_pwr)
            else:
                engine_adj = dc.soc_error[i] * max_regen_pwr * ess_soc_gain

            dc.loc[i, "engine_adj_kw"] = engine_adj

        else:
            shift = i - 1

            if (dc.fc_priority[i] == 1):
                if (dc.ess_priority[i] == 0):
                    fc_pwr_req = dc.elec_pwr_out_did[i] / pwr_conv_eff
                elif ((dc.elec_pwr_out_did[i] - dc.engine_adj_kw[shift]) > 0):
                    fc_pwr_req = (dc.elec_pwr_out_did[i] - dc.engine_adj_kw[shift]) / pwr_conv_eff
                elif ((dc.elec_pwr_out_did[i] - dc.engine_adj_kw[shift]) < 0):
                    fc_pwr_req = 0
                else:
                    fc_pwr_req = (dc.elec_pwr_out_did[i] - dc.engine_adj_kw[shift]) / pwr_conv_eff
            else:
                fc_pwr_req = 0

            dc.loc[i, "fc_pwr_req"] = fc_pwr_req
            dc.loc[i, "fc_pwr_conv_loss"] = dc.fc_pwr_req[i] * (1 - pwr_conv_eff)
            dc.loc[i, "fc_step_up"] = dc.time_step[i] * fc_pwr_max / fc_time_full
            dc.loc[i, "fc_step_down"] = dc.time_step[i] * fc_pwr_max / fc_time_zero

            if ((dc.fc_pwr_req[i] - dc.fc_pwr_out[shift]) >= 0):
                if ((dc.fc_pwr_req[i] - dc.fc_pwr_out[shift]) > dc.fc_step_up[i]):
                    fc_pwr_ramp = dc.fc_pwr_out[shift] + dc.fc_step_up[i]
                else:
                    fc_pwr_ramp = dc.fc_pwr_out[shift] + (dc.fc_pwr_req[i] - dc.fc_pwr_out[shift])
            elif (abs(dc.fc_pwr_req[i] - dc.fc_pwr_out[shift]) > dc.fc_step_down[i]):
                fc_pwr_ramp = dc.fc_pwr_out[shift] - dc.fc_step_down[i]
            else:
                fc_pwr_ramp = dc.fc_pwr_out[shift] + (dc.fc_pwr_req[i] - dc.fc_pwr_out[shift])

            dc.loc[i, "fc_pwr_ramp"] = fc_pwr_ramp

            if (dc.fc_priority[i] == 1):
                if (fc_pwr_max >= dc.fc_pwr_ramp[i]):
                    fc_pwr_out = dc.fc_pwr_ramp[i]
                else:
                    fc_pwr_out = fc_pwr_max
            else:
                fc_pwr_out = 0

            dc.loc[i, "fc_pwr_out"] = fc_pwr_out

            if (dc.fc_priority[i] == 1):
                fc_pwr_clip = dc.fc_pwr_req[i] - dc.fc_pwr_out[i]
            else:
                fc_pwr_clip = 0

            dc.loc[i, "fc_pwr_clip"] = fc_pwr_clip
            dc.loc[i, "fc_pwr_deficit"] = dc.elec_pwr_out[i] - dc.fc_pwr_out[i]

            ### ESS DISCHARGE

            # ess_pwr_dis_req conditions
            ess_m1 = (dc.ic_priority[i] == 1) & (dc.ess_priority[i] > 0)
            ess_m2 = ((dc.fc_priority[i] == 1) & (dc.ess_priority[i] > 0)) & (
                (dc.elec_pwr_out_did[i] - dc.fc_pwr_out[i]) <= 0)
            ess_m3 = ((dc.fc_priority[i] == 1) & (dc.ess_priority[i] > 0))
            ess_m4 = (dc.ess_priority[i] == 1)
            ess_pwr_dis_req_conditions = [ess_m1, ess_m2, ess_m3, ess_m4]

            # ess_pwr_dis_req values
            ess_v1 = (dc.elec_pwr_out_did[i] / pwr_conv_eff)
            ess_v2 = 0
            ess_v3 = ((dc.elec_pwr_out_did[i] - dc.fc_pwr_out[i]) / pwr_conv_eff)
            ess_v4 = (dc.elec_pwr_out_did[i] / pwr_conv_eff)

            ess_pwr_dis_req_vals = [ess_v1, ess_v2, ess_v3, ess_v4]

            dc.loc[i, "ess_pwr_dis_req"] = np.select(ess_pwr_dis_req_conditions, ess_pwr_dis_req_vals, default=0)
            dc.loc[i, "ess_pwr_conv_dis_loss"] = dc.ess_pwr_dis_req[i] * (1 - pwr_conv_eff)
            dc.loc[i, "ess_max_dis_step"] = dc.time_step[i] * ess_pwr_max / ess_time_full

            if (dc.ess_pwr_dis_req[i] - dc.ess_pwr_dis_out[shift]) >= 0:
                if (dc.ess_pwr_dis_req[i] - dc.ess_pwr_dis_out[shift]) > dc.ess_max_dis_step[i]:
                    ess_pwr_dis_ramp = dc.ess_pwr_dis_out[shift] + dc.ess_max_dis_step[i]
                else:
                    ess_pwr_dis_ramp = dc.ess_pwr_dis_out[shift] + (dc.ess_pwr_dis_req[i] - dc.ess_pwr_dis_out[shift])
            elif (dc.ess_pwr_dis_req[i] >= dc.ess_pwr_dis_out[shift]):
                ess_pwr_dis_ramp = dc.ess_pwr_dis_out[shift]
            else:
                ess_pwr_dis_ramp = dc.ess_pwr_dis_req[i]

            dc.loc[i, "ess_pwr_dis_ramp"] = ess_pwr_dis_ramp

            dc.loc[i, "ess_pwr_dis_gross"] = dc.ess_pwr_dis_ramp[i] / np.sqrt(ess_round_trip_eff)
            dc.loc[i, "ess_pwr_dis_loss"] = np.abs(dc.ess_pwr_dis_gross[i]) * (1 - np.sqrt(ess_round_trip_eff))
            dc.loc[i, "ess_soc_lim"] = np.where(dc.ess_soc[shift] <= ess_min_soc, 0, dc.ess_pwr_dis_gross[i])
            dc.loc[i, "ess_pwr_dis_out"] = np.where(ess_pwr_max >= dc.ess_soc_lim[i], dc.ess_soc_lim[i], ess_pwr_max)
            dc.loc[i, "ess_pwr_dis_clip"] = dc.ess_pwr_dis_gross[i] - dc.ess_pwr_dis_out[i]

            if ((dc.ic_priority[i] == 1) & (dc.ess_priority[i] > 0)):
                ess_pwr_chrg_req = dc.elec_pwr_out_chrg[i] / pwr_conv_eff
            elif (((dc.fc_priority[i] == 1) & (dc.ess_priority[i] > 0)) & (dc.fc_pwr_deficit[i] <= 0)):
                ess_pwr_chrg_req = dc.fc_pwr_deficit[i] / pwr_conv_eff
            elif (dc.ess_priority[i] == 1):
                ess_pwr_chrg_req = dc.elec_pwr_out_chrg[i] / pwr_conv_eff
            else:
                ess_pwr_chrg_req = 0

            dc.loc[i, "ess_pwr_chrg_req"] = ess_pwr_chrg_req
            dc.loc[i, "ess_pwr_conv_chrg_loss"] = dc.ess_pwr_chrg_req[i] * (1 - pwr_conv_eff)
            dc.loc[i, "ess_pwr_chrg_net"] = dc.ess_pwr_chrg_req[i] / (sqrt(ess_round_trip_eff))

            ###Calculate the map values without needing the regen map
            dc.loc[i, "regen_pcnt"] = regen_eff_max / (
                1 + regen_a * np.exp((-1 * regen_b) * (dc.speed[i] * KPH_TO_MPH + 1)))
            dc.loc[i, "ess_pwr_regen_map"] = dc.regen_pcnt[i] * dc.ess_pwr_chrg_net[i]

            if (dc.ess_soc[shift] >= ess_max_soc):
                ess_pwr_soc_lim = 0
            else:
                ess_pwr_soc_lim = dc.ess_pwr_regen_map[i]

            dc.loc[i, "ess_pwr_soc_lim"] = ess_pwr_soc_lim

            if (dc.ess_priority[i] > 0 & (max_regen_pwr >= abs(dc.ess_pwr_soc_lim[i]))):
                ess_pwr_chrg_out = dc.ess_pwr_soc_lim[i]
            elif (dc.ess_priority[i] > 0):
                ess_pwr_chrg_out = (-1) * max_regen_pwr
            else:
                ess_pwr_chrg_out = 0

            dc.loc[i, "ess_pwr_chrg_out"] = ess_pwr_chrg_out
            dc.loc[i, "ess_pwr_chrg_loss"] = abs(dc.ess_pwr_chrg_out[i]) * (1 - sqrt(ess_round_trip_eff))
            dc.loc[i, "ess_pwr_chrg_clip"] = dc.ess_pwr_chrg_net[i] - dc.ess_pwr_chrg_out[i]
            dc.loc[i, "mech_brk_pwr"] = dc.pwr_trans_deccel[i] - dc.ess_pwr_chrg_out[i]

            dc.loc[i, "ess_cur_cap"] = dc.ess_cur_cap[shift] - (dc.ess_pwr_dis_out[i] + dc.ess_pwr_chrg_out[i]) * \
                                       dc.time_step[i] / 3600
            dc.loc[i, "ess_soc"] = dc.ess_cur_cap[i] / ess_cap * 100
            dc.loc[i, "soc_error"] = dc.ess_soc[i] - ess_target_soc  # * 100

            if ((dc.soc_error[i] * ess_pwr_max * ess_soc_gain) >= max_regen_pwr):
                engine_adj = max_regen_pwr
            elif (dc.soc_error[i] * max_regen_pwr * ess_soc_gain <= ((-1) * max_regen_pwr)):
                engine_adj = ((-1) * max_regen_pwr)
            else:
                engine_adj = dc.soc_error[i] * max_regen_pwr * ess_soc_gain  # *100

            dc.loc[
                i, "engine_adj_kw"] = engine_adj  # np.select( engine_adj_kw_conditions, engine_adj_kw_vals, default = eak_default )

    return dc


def ess_requirements_beb(dc, ess_pwr_max, ess_init_soc, ess_cap, fc_pwr_max, pwr_conv_eff, fc_time_full, fc_time_zero,
                         ess_round_trip_eff, ess_time_full, regen_eff_max, regen_a, regen_b, max_regen_pwr,
                         ess_soc_gain, ess_target_soc, ess_min_soc, ess_max_soc):
    from math import sqrt
    KPH_TO_MPH = 0.621371
    dc = dc.reset_index(drop=True)

    dc["ess_pwr_dis_req"] = 0
    dc["ess_pwr_conv_dis_loss"] = 0
    dc["ess_max_dis_step"] = 0
    dc["ess_pwr_dis_ramp"] = 0
    dc["ess_pwr_dis_gross"] = 0
    dc["ess_pwr_dis_loss"] = 0
    dc["ess_pwr_dis_clip"] = 0
    dc["ess_soc_lim"] = 0
    dc["ess_pwr_dis_out"] = ess_pwr_max

    dc["ess_pwr_chrg_req"] = 0
    dc["ess_pwr_conv_chrg_loss"] = 0
    dc["ess_pwr_chrg_net"] = 0
    dc["ess_pwr_regen_map"] = 0
    dc["ess_pwr_soc_lim"] = 0
    dc["ess_pwr_chrg_out"] = 0
    dc["ess_pwr_chrg_loss"] = 0
    dc["ess_pwr_chrg_clip"] = 0
    dc["mech_brk_pwr"] = 0

    dc["ess_soc"] = ess_init_soc
    dc["ess_cur_cap"] = ess_cap * ess_init_soc / 100
    dc["soc_error"] = 0
    dc["engine_adj_kw"] = 0

    # dc["ess_cur_cap_dis"] =  0

    dc["regen_pcnt"] = 0.005247

    for i, row in dc.iterrows():

        if i == 0:

            epcr_m1 = (dc.ic_priority[i] == 1) & (dc.ess_priority[i] > 0)
            epcr_m2 = ((dc.fc_priority[i] == 1) & (dc.ess_priority[i] > 0)) & (dc.fc_pwr_deficit[i] <= 0)
            epcr_m3 = (dc.ess_priority[i] == 1)
            ess_pwr_chrg_req_conditions = [epcr_m1, epcr_m2, epcr_m3]

            # ess_pwr_chrg_req values
            epcr_v1 = dc.elec_pwr_out_chrg[i] / pwr_conv_eff
            epcr_v2 = dc.fc_pwr_deficit[i] / pwr_conv_eff
            epcr_v3 = dc.elec_pwr_out_chrg[i] / pwr_conv_eff
            ess_pwr_chrg_req_vals = [epcr_v1, epcr_v2, epcr_v3]

            dc.loc[i, "ess_pwr_chrg_req"] = np.select(ess_pwr_chrg_req_conditions, ess_pwr_chrg_req_vals, default=0)
            dc.loc[i, "ess_pwr_chrg_net"] = dc.ess_pwr_chrg_req[i] / (np.sqrt(ess_round_trip_eff))
            dc.loc[i, "ess_pwr_conv_chrg_loss"] = dc.ess_pwr_chrg_req[i] * (1 - pwr_conv_eff)

            ess_m1 = (dc.ic_priority[i] == 1) & (dc.ess_priority[i] > 0)
            ess_m2 = ((dc.fc_priority[i] == 1) & (dc.ess_priority[i] > 0)) & (
                (dc.elec_pwr_out_did[i] - dc.fc_pwr_out[i]) <= 0)
            ess_m3 = ((dc.fc_priority[i] == 1) & (dc.ess_priority[i] > 0))
            ess_m4 = (dc.ess_priority[i] == 1)
            ess_pwr_dis_req_conditions = [ess_m1, ess_m2, ess_m3, ess_m4]

            # ess_pwr_dis_req values
            ess_v1 = (dc.elec_pwr_out_did[i] / pwr_conv_eff)
            ess_v2 = 0
            ess_v3 = ((dc.elec_pwr_out_did[i] - dc.fc_pwr_out[i]) / pwr_conv_eff)
            ess_v4 = (dc.elec_pwr_out_did[i] / pwr_conv_eff)

            ess_pwr_dis_req_vals = [ess_v1, ess_v2, ess_v3, ess_v4]

            dc.loc[i, "ess_pwr_dis_req"] = np.select(ess_pwr_dis_req_conditions, ess_pwr_dis_req_vals, default=0)
            dc.loc[i, "ess_pwr_conv_dis_loss"] = dc.ess_pwr_dis_req[i] * (1 - pwr_conv_eff)
            dc.loc[i, "ess_max_dis_step"] = dc.time_step[i] * ess_pwr_max / ess_time_full

            dc.loc[i, "ess_pwr_dis_gross"] = dc.ess_pwr_dis_ramp[i] / np.sqrt(ess_round_trip_eff)
            dc.loc[i, "ess_pwr_dis_loss"] = np.abs(dc.ess_pwr_dis_gross[i]) * (1 - np.sqrt(ess_round_trip_eff))

            dc.loc[i, "regen_pcnt"] = regen_eff_max / (
                1 + regen_a * np.exp((-1 * regen_b) * (dc.speed[i] * KPH_TO_MPH + 1)))

            dc.loc[i, "ess_pwr_regen_map"] = dc.regen_pcnt[i] * dc.ess_pwr_chrg_net[i]

            dc.loc[i, "ess_pwr_dis_out"] = np.where(ess_pwr_max >= dc.ess_soc_lim[i], dc.ess_soc_lim[i], ess_pwr_max)

            dc.loc[i, "ess_pwr_soc_lim"] = np.where(dc.ess_pwr_regen_map[i] < 0, dc.ess_pwr_regen_map[i], 0)
            # dc["ess_pwr_soc_lim"] =  dc.ess_pwr_regen_map
            dc.loc[i, "ess_pwr_chrg_out"] = np.where(
                ((dc.ess_priority[i] > 0) & (max_regen_pwr >= np.abs(dc.ess_pwr_soc_lim[i]))), dc.ess_pwr_soc_lim[i], 0)

            dc.loc[i, "ess_pwr_dis_clip"] = dc.ess_pwr_dis_gross[i] - dc.ess_pwr_dis_out[i]

            dc.loc[i, "mech_brk_pwr"] = dc.pwr_trans_deccel[i] - dc.ess_pwr_chrg_out[i]
            dc.loc[i, "ess_pwr_chrg_clip"] = dc.ess_pwr_chrg_net[i] - dc.ess_pwr_chrg_out[i]
            dc.loc[i, "ess_pwr_chrg_loss"] = abs(dc.ess_pwr_chrg_out[i]) * (1 - sqrt(ess_round_trip_eff))

            # dc.loc[ i, "ess_cur_cap_dis"] =  (dc.ess_pwr_chrg_out[i] + dc.ess_pwr_dis_out[i] ) * dc.time_step[i] / 3600

            dc.loc[i, "ess_soc_lim"] = np.where(dc.ess_cur_cap[i] / ess_cap <= ess_min_soc, 0, dc.ess_pwr_dis_gross[i])
            dc.loc[i, "ess_pwr_dis_out"] = np.where(ess_pwr_max >= dc.ess_soc_lim[i], dc.ess_soc_lim[i], ess_pwr_max)

            dc.loc[i, "ess_soc"] = ess_init_soc

            dc.loc[i, "ess_cur_cap"] = ess_cap * ess_init_soc / 100

            # Error in SOC vs Target
            dc.loc[i, "soc_error"] = dc.ess_soc[i] - ess_target_soc

            # engine_adj_kw conditions ~~~~in the model the SOC error is not percent, scale by 100 ~~~~~~~~~~~~~~~~~~~~VERIFY THAT THIS LOGIC IS CORRECT ~~~~~~~~~~~~~~~~~
            eak_m1 = dc.soc_error[i] * ess_pwr_max * ess_soc_gain >= max_regen_pwr
            eak_m2 = dc.soc_error[i] * max_regen_pwr * ess_soc_gain <= ((-1) * max_regen_pwr)
            engine_adj_kw_conditions = [eak_m1, eak_m2]

            # engine_adj_kw values
            eak_v1 = max_regen_pwr
            eak_v2 = (-1) * max_regen_pwr
            eak_default = dc.soc_error[i] * max_regen_pwr * ess_soc_gain
            engine_adj_kw_vals = [eak_v1, eak_v2]

            # Engine Adjust Signal to Balance SOC
            # dc.loc[ i, "engine_adj_kw"] = np.select( engine_adj_kw_conditions, engine_adj_kw_vals, default = eak_default )
            if dc.soc_error[i] * ess_pwr_max * ess_soc_gain >= max_regen_pwr:
                engine_adj = max_regen_pwr
            elif dc.soc_error[i] * max_regen_pwr * ess_soc_gain <= ((-1) * max_regen_pwr):
                engine_adj = ((-1) * max_regen_pwr)
            else:
                engine_adj = dc.soc_error[i] * max_regen_pwr * ess_soc_gain

            dc.loc[i, "engine_adj_kw"] = engine_adj

        else:
            shift = i - 1

            ### ESS DISCHARGE

            # ess_pwr_dis_req conditions
            ess_m1 = (dc.ic_priority[i] == 1) & (dc.ess_priority[i] > 0)
            ess_m2 = ((dc.fc_priority[i] == 1) & (dc.ess_priority[i] > 0)) & (
                (dc.elec_pwr_out_did[i] - dc.fc_pwr_out[i]) <= 0)
            ess_m3 = ((dc.fc_priority[i] == 1) & (dc.ess_priority[i] > 0))
            ess_m4 = (dc.ess_priority[i] == 1)
            ess_pwr_dis_req_conditions = [ess_m1, ess_m2, ess_m3, ess_m4]

            # ess_pwr_dis_req values
            ess_v1 = (dc.elec_pwr_out_did[i] / pwr_conv_eff)
            ess_v2 = 0
            ess_v3 = ((dc.elec_pwr_out_did[i] - dc.fc_pwr_out[i]) / pwr_conv_eff)
            ess_v4 = (dc.elec_pwr_out_did[i] / pwr_conv_eff)

            ess_pwr_dis_req_vals = [ess_v1, ess_v2, ess_v3, ess_v4]

            dc.loc[i, "ess_pwr_dis_req"] = np.select(ess_pwr_dis_req_conditions, ess_pwr_dis_req_vals, default=0)
            dc.loc[i, "ess_pwr_conv_dis_loss"] = dc.ess_pwr_dis_req[i] * (1 - pwr_conv_eff)
            dc.loc[i, "ess_max_dis_step"] = dc.time_step[i] * ess_pwr_max / ess_time_full

            m1 = ((dc.ess_pwr_dis_req[i] - dc.ess_pwr_dis_out[shift]) >= 0) & ((dc.ess_pwr_dis_req[i] - dc.ess_pwr_dis_out[shift]) > dc.ess_max_dis_step[i])
            m2 = ((dc.ess_pwr_dis_req[i] - dc.ess_pwr_dis_out[shift]) >= 0)
            m3 = (dc.ess_pwr_dis_req[i] >= dc.ess_pwr_dis_out[shift])
            ess_pwr_dis_ramp_conds = [m1, m2, m3]

            v1 = dc.ess_pwr_dis_out[shift] + dc.ess_max_dis_step[i]
            v2 = dc.ess_pwr_dis_out[shift] + (dc.ess_pwr_dis_req[i] - dc.ess_pwr_dis_out[shift])
            v3 = dc.ess_pwr_dis_out[shift]
            ess_pwr_dis_ramp_vals = [v1, v2, v3]

            # if (dc.ess_pwr_dis_req[i] - dc.ess_pwr_dis_out[shift]) >= 0:
            #     if (dc.ess_pwr_dis_req[i] - dc.ess_pwr_dis_out[shift]) > dc.ess_max_dis_step[i]:
            #         ess_pwr_dis_ramp = dc.ess_pwr_dis_out[shift] + dc.ess_max_dis_step[i]
            #     else:
            #         ess_pwr_dis_ramp = dc.ess_pwr_dis_out[shift] + (dc.ess_pwr_dis_req[i] - dc.ess_pwr_dis_out[shift])
            # elif (dc.ess_pwr_dis_req[i] >= dc.ess_pwr_dis_out[shift]):
            #     ess_pwr_dis_ramp = dc.ess_pwr_dis_out[shift]
            # else:
            #     ess_pwr_dis_ramp =  dc.ess_pwr_dis_out[shift]

            dc.loc[i, "ess_pwr_dis_ramp"] = np.select(ess_pwr_dis_ramp_conds, ess_pwr_dis_ramp_vals, default=dc.ess_pwr_dis_out[shift])

            dc.loc[i, "ess_pwr_dis_gross"] = dc.ess_pwr_dis_ramp[i] / np.sqrt(ess_round_trip_eff)
            dc.loc[i, "ess_pwr_dis_loss"] = np.abs(dc.ess_pwr_dis_gross[i]) * (1 - np.sqrt(ess_round_trip_eff))
            dc.loc[i, "ess_soc_lim"] = np.where(dc.ess_soc[shift] <= ess_min_soc, 0, dc.ess_pwr_dis_gross[i])
            dc.loc[i, "ess_pwr_dis_out"] = np.where(ess_pwr_max >= dc.ess_soc_lim[i], dc.ess_soc_lim[i], ess_pwr_max)
            dc.loc[i, "ess_pwr_dis_clip"] = dc.ess_pwr_dis_gross[i] - dc.ess_pwr_dis_out[i]

            # if ((dc.ic_priority[i] == 1) & (dc.ess_priority[i] > 0)):
            #     ess_pwr_chrg_req = dc.elec_pwr_out_chrg[i] / pwr_conv_eff
            # elif (((dc.fc_priority[i] == 1) & (dc.ess_priority[i] > 0)) & (dc.fc_pwr_deficit[i] <= 0)):
            #     ess_pwr_chrg_req = dc.fc_pwr_deficit[i] / pwr_conv_eff
            # elif (dc.ess_priority[i] == 1):
            #     ess_pwr_chrg_req = dc.elec_pwr_out_chrg[i] / pwr_conv_eff
            # else:
            #     ess_pwr_chrg_req = 0
            m1 = ((dc.ic_priority[i] == 1) & (dc.ess_priority[i] > 0))
            m2 = (((dc.fc_priority[i] == 1) & (dc.ess_priority[i] > 0)) & (dc.fc_pwr_deficit[i] <= 0))
            m3 = (dc.ess_priority[i] == 1)
            v1 = dc.elec_pwr_out_chrg[i] / pwr_conv_eff
            v2 = dc.fc_pwr_deficit[i] / pwr_conv_eff
            v3 = dc.elec_pwr_out_chrg[i] / pwr_conv_eff
            ess_pwr_chrg_req_conditions = [m1, m2, m3]
            ess_pwr_chrg_req_vals = [v1, v2, v3]
            
            dc.loc[i, "ess_pwr_chrg_req"] = np.select(ess_pwr_chrg_req_conditions, ess_pwr_chrg_req_vals, default=0)
            dc.loc[i, "ess_pwr_conv_chrg_loss"] = dc.ess_pwr_chrg_req[i] * (1 - pwr_conv_eff)
            dc.loc[i, "ess_pwr_chrg_net"] = dc.ess_pwr_chrg_req[i] / (sqrt(ess_round_trip_eff))

            ###Calculate the map values without needing the regen map
            dc.loc[i, "regen_pcnt"] = regen_eff_max / (
                1 + regen_a * np.exp((-1 * regen_b) * (dc.speed[i] * KPH_TO_MPH + 1)))
            dc.loc[i, "ess_pwr_regen_map"] = dc.regen_pcnt[i] * dc.ess_pwr_chrg_net[i]

            if (dc.ess_soc[shift] >= ess_max_soc):
                ess_pwr_soc_lim = 0
            else:
                ess_pwr_soc_lim = dc.ess_pwr_regen_map[i]

            dc.loc[i, "ess_pwr_soc_lim"] = ess_pwr_soc_lim

            if (dc.ess_priority[i] > 0 & (max_regen_pwr >= abs(dc.ess_pwr_soc_lim[i]))):
                ess_pwr_chrg_out = dc.ess_pwr_soc_lim[i]
            elif (dc.ess_priority[i] > 0):
                ess_pwr_chrg_out = (-1) * max_regen_pwr
            else:
                ess_pwr_chrg_out = 0

            dc.loc[i, "ess_pwr_chrg_out"] = ess_pwr_chrg_out
            dc.loc[i, "ess_pwr_chrg_loss"] = abs(dc.ess_pwr_chrg_out[i]) * (1 - sqrt(ess_round_trip_eff))
            dc.loc[i, "ess_pwr_chrg_clip"] = dc.ess_pwr_chrg_net[i] - dc.ess_pwr_chrg_out[i]
            dc.loc[i, "mech_brk_pwr"] = dc.pwr_trans_deccel[i] - dc.ess_pwr_chrg_out[i]

            dc.loc[i, "ess_cur_cap"] = dc.ess_cur_cap[shift] - (dc.ess_pwr_dis_out[i] + dc.ess_pwr_chrg_out[i]) * \
                                       dc.time_step[i] / 3600
            dc.loc[i, "ess_soc"] = dc.ess_cur_cap[i] / ess_cap * 100
            dc.loc[i, "soc_error"] = dc.ess_soc[i] - ess_target_soc  # * 100

            if ((dc.soc_error[i] * ess_pwr_max * ess_soc_gain) >= max_regen_pwr):
                engine_adj = max_regen_pwr
            elif (dc.soc_error[i] * max_regen_pwr * ess_soc_gain <= ((-1) * max_regen_pwr)):
                engine_adj = ((-1) * max_regen_pwr)
            else:
                engine_adj = dc.soc_error[i] * max_regen_pwr * ess_soc_gain  # *100

            dc.loc[
                i, "engine_adj_kw"] = engine_adj  # np.select( engine_adj_kw_conditions, engine_adj_kw_vals, default = eak_default )

    return dc


def hvac_heating(engine_type, temperature):
    col_names = ['Heating', 'Cooling', 'Unit']

    HOURS_PER_DAY = 24

    ### Set bus type here
    BUS_TYPE = engine_type

    parameters_list = {
        ### General Parameters
        'General Parameters':
            {'Outside Temperature': [temperature, temperature, 'C'],
             "Temperature Setpoint": [22.2, 22.2, 'C'],
             "Minimum HVAC Load": [2, 2, 'kW'],
             'Bus Window Area': [29, 29, 'm2'],
             'Average Speed': [40, 40, 'km/h'],
             'Air Density': [1.27, 1.27, 'kg/m3'],
             'Air Specific Heat': [1009, 1009, 'J/kg*K'],
             'Outside Air Convection Coefficient': [26.2, 26.15, 'W/m2*K'],  ### Calculated - Placeholder
             'Inside Air Convection Coefficient': [9, 9, 'W/m2*K'],  ### Calculated - Placeholder
             },
        ### Heat from Passengers
        'Heat from Passengers':
            {'Heat Output per Person': [60, 85, 'W/m2'],
             'Person surface area': [1.8, 1.8, 'm2'],
             'Passengers per bus': [40, 40, 'people'], },

        ### Heat from the Sun
        'Heat from the Sun':
            {'Min/Max Mean Daily Global Insolation': [0.71, 6.12, 'kWh/m2'],
             'Hours per Day': [24, 24, 'hours/day'],
             'Min/Max Solar Irradiation': [29.58, 255.00, 'W/m2'],  ### Calculated - Placeholder
             'Min/Max Temperature': [-40.00, 35.00, 'C'],
             'Solar Irradiation': [29.58, 255.00, 'W/m2'],
             'Window Transmissivity': [0.75, 0.75, 'K'], },

        ### Heat from Necessary Venitlation
        'Heat from Necessary Venitlation':
            {'Air Flow Rate': [0.05, 0.05, 'm3/s'], },

        ### Heat through Windows
        'Heat through Windows':
            {'Window Thickness': [0.006, 0.006, 'm'],
             'Window conductivity': [1.05, 1.05, 'W/m*K'],
             'Window heat transfer coefficient': [6.45, 6.45, 'W/m2*K'], },  ### Calculated - Placeholder

        ### Heat through Walls
        'Heat through Walls':
            {'Wall Heat Transfer Coefficient': [2.283, 2.283, 'W/m2*k'],
             'Wall Area': [27, 27, 'm2'], },

        ### Heat through Roof
        'Heat through Roof':
            {'Roof Heat Transfer Coefficient': [1.87, 1.87, 'W/m2*K'],
             'Roof Area': [29, 29, 'm2'], },

        ### Heat Loss Floor
        'Heat Loss Floor':
            {'Floor Heat Transfer Coefficient': [2.747, 2.747, 'W/m2*K'],
             'Floor Area': [29.43, 29.43, 'm2'],
             },

        ### Heat through Door when open
        'Heat through Door when open':
            {'Volumetric Flow Rate': [0.4, 0.4, 'm3/s'],
             'Heat loss per door open': [19.08, -6.55, 'kW/door/open'],
             'Stop Time': [0.5, 0.5, 'minutes/stop'],
             'Time between stops': [4, 4, 'minutes/drive'],
             'Doors per bus': [3, 3, 'doors/bus'], },

        ### Fuel Cell Heat
        'Fuel Cell Heat':
            {'Fuel cell heat output': [15.0, 0, 'kW'],
             'Max percent of total heat': [100, 0, '%'], },

        ### HVAC Required
        'HVAC Required': {'HVAC Efficiency (EER/COP)': [1, 2.2, '-']},
    }

    reformed_dict = {}
    for outerKey, innerDict in parameters_list.items():
        for innerKey, values in innerDict.items():
            reformed_dict[(outerKey,
                           innerKey)] = values

    parameters = pd.DataFrame.from_dict(reformed_dict, orient='index', columns=col_names)
    index = pd.MultiIndex.from_tuples(parameters.index, names=('Category', 'Parameter'))
    parameters = pd.DataFrame(parameters, index=index)

    ### cols to variables
    outside_temp_h, outside_temp_c = parameters.loc[('General Parameters', 'Outside Temperature'), :'Cooling']
    temp_set_h, temp_set_c = parameters.loc[('General Parameters', 'Temperature Setpoint'), :'Cooling']
    min_hvac_load_h, min_hvac_load_c = parameters.loc[('General Parameters', 'Minimum HVAC Load'), :'Cooling']
    bus_window_area, _ = parameters.loc[('General Parameters', 'Bus Window Area'), :'Cooling']
    ave_speed_h, ave_speed_c = parameters.loc[('General Parameters', 'Average Speed'), :'Cooling']
    air_density_h, air_density_c = parameters.loc[('General Parameters', 'Air Density'), :'Cooling']
    air_specific_heat = parameters.loc[('General Parameters', 'Air Specific Heat'), 'Heating']
    outside_air_coeff_h, outside_air_coeff_c = parameters.loc[
                                               ('General Parameters', 'Outside Air Convection Coefficient'), :'Cooling']
    inside_air_coeff_h, inside_air_coeff_c = parameters.loc[('General Parameters', 'Inside Air Convection Coefficient'),
                                             :'Cooling']

    heat_output_pp_h, heat_output_pp_c = parameters.loc[('Heat from Passengers', 'Heat Output per Person'), :'Cooling']
    person_surface_area = parameters.loc[('Heat from Passengers', 'Person surface area'), 'Cooling']
    passengers_per_bus = parameters.loc[('Heat from Passengers', 'Passengers per bus'), 'Cooling']

    daily_global_insul_h, daily_global_insul_c = parameters.loc[
                                                 ('Heat from the Sun', 'Min/Max Mean Daily Global Insolation'),
                                                 :'Cooling']
    min_max_solar_irradiation_h, min_max_solar_irradiation_c = parameters.loc[
                                                               ('Heat from the Sun', 'Min/Max Solar Irradiation'),
                                                               :'Cooling']
    min_max_temp_h, min_max_temp_c = parameters.loc[('Heat from the Sun', 'Min/Max Temperature'), :'Cooling']
    solar_irradiation_h, solar_irradiation_c = parameters.loc[('Heat from the Sun', 'Solar Irradiation'), :'Cooling']
    window_tranmissivity, _ = parameters.loc[('Heat from the Sun', 'Window Transmissivity'), :'Cooling']

    air_flow_rate, _ = parameters.loc[('Heat from Necessary Venitlation', 'Air Flow Rate'), :'Cooling']
    window_thickness, _ = parameters.loc[('Heat through Windows', 'Window Thickness'), :'Cooling']
    window_conductivity, _ = parameters.loc[('Heat through Windows', 'Window conductivity'), :'Cooling']
    window_ht_coeff_h, window_ht_coeff_c = parameters.loc[('Heat through Windows', 'Window heat transfer coefficient'),
                                           :'Cooling']
    wall_ht_coeff, _ = parameters.loc[('Heat through Walls', 'Wall Heat Transfer Coefficient'), :'Cooling']
    wall_area, _ = parameters.loc[('Heat through Walls', 'Wall Area'), :'Cooling']
    roof_ht_coeff, _ = parameters.loc[('Heat through Roof', 'Roof Heat Transfer Coefficient'), :'Cooling']
    roof_area, _ = parameters.loc[('Heat through Roof', 'Roof Area'), :'Cooling']
    floor_ht_coeff, _ = parameters.loc[('Heat Loss Floor', 'Floor Heat Transfer Coefficient'), :'Cooling']
    floor_area, _ = parameters.loc[('Heat Loss Floor', 'Floor Area'), :'Cooling']

    vol_flow_rate, _ = parameters.loc[('Heat through Door when open', 'Volumetric Flow Rate'), :'Cooling']
    heat_loss_door_open_h, heat_loss_door_open_c = parameters.loc[
                                                   ('Heat through Door when open', 'Heat loss per door open'),
                                                   :'Cooling']
    stop_time, _ = parameters.loc[('Heat through Door when open', 'Stop Time'), :'Cooling']
    time_between_stops, _ = parameters.loc[('Heat through Door when open', 'Time between stops'), :'Cooling']
    doors_per_bus, _ = parameters.loc[('Heat through Door when open', 'Doors per bus'), :'Cooling']
    windows_coeff_h = parameters.loc[('Heat through Windows', 'Window Thickness'), 'Heating'] / parameters.loc[
        ('Heat through Windows', 'Window conductivity'), 'Heating']
    windows_coeff_c = parameters.loc[('Heat through Windows', 'Window Thickness'), 'Cooling'] / parameters.loc[
        ('Heat through Windows', 'Window conductivity'), 'Cooling']

    fuel_cell_heat_out_h, fuel_cell_heat_out_c = parameters.loc[('Fuel Cell Heat', 'Fuel cell heat output'), :'Cooling']
    max_pcnt_total_heat_h, max_pcnt_total_heat_c = parameters.loc[('Fuel Cell Heat', 'Max percent of total heat'),
                                                   :'Cooling']

    outside_air_coeff_h = 9 + 3.5 * (ave_speed_h / 3.6) ** (0.66)
    outside_air_coeff_c = 9 + 3.5 * (ave_speed_c / 3.6) ** (0.66)

    min_max_solar_irradiation_h = daily_global_insul_h / HOURS_PER_DAY * 1000
    min_max_solar_irradiation_c = daily_global_insul_c / HOURS_PER_DAY * 1000

    solar_irradiation_h = min_max_solar_irradiation_h + (min_max_solar_irradiation_c - min_max_solar_irradiation_h) * (
        outside_temp_h - min_max_temp_h) / (min_max_temp_c - min_max_temp_h)
    solar_irradiation_c = min_max_solar_irradiation_h + (min_max_solar_irradiation_c - min_max_solar_irradiation_h) * (
        outside_temp_c - min_max_temp_h) / (min_max_temp_c - min_max_temp_h)

    window_ht_coeff_h = 1 / (1 / outside_air_coeff_h + 1 / inside_air_coeff_h + windows_coeff_h)
    window_ht_coeff_c = 1 / (1 / outside_air_coeff_c + 1 / inside_air_coeff_c + windows_coeff_c)

    heat_loss_door_open_h = vol_flow_rate * air_density_h * air_specific_heat * (temp_set_h - outside_temp_h) / 1000
    heat_loss_door_open_c = vol_flow_rate * air_density_c * air_specific_heat * (temp_set_h - outside_temp_h) / 1000

    ### Assign Values to variables
    qlist = ['q_pas', 'q_sun', 'q_ven', 'q_window', 'q_wall', 'q_roof', 'q_floor', 'q_door', 'q_fc',
             'q_hvac', ]  # 'HVAC Efficiency', 'HVAC Energy Requirement', 'Total'
    cols = ['Heating', 'Cooling']

    q_heating = pd.DataFrame(index=qlist, columns=cols)

    q_heating.loc['q_pas', 'Heating'] = heat_output_pp_h * person_surface_area * passengers_per_bus / 1000
    q_heating.loc['q_pas', 'Cooling'] = heat_output_pp_c * person_surface_area * passengers_per_bus / 1000
    q_heating.loc['q_sun', 'Heating'] = bus_window_area * solar_irradiation_h * window_tranmissivity / 1000
    q_heating.loc['q_sun', 'Cooling'] = bus_window_area * solar_irradiation_c * window_tranmissivity / 1000
    q_heating.loc['q_ven', 'Heating'] = (-1) * (
        temp_set_h - outside_temp_h) * air_density_h * air_specific_heat * air_flow_rate / 1000
    q_heating.loc['q_ven', 'Cooling'] = (-1) * (
        temp_set_c - outside_temp_c) * air_density_c * air_specific_heat * air_flow_rate / 1000
    q_heating.loc['q_window', 'Heating'] = (-1) * (
        temp_set_h - outside_temp_h) * bus_window_area * window_ht_coeff_h / 1000
    q_heating.loc['q_window', 'Cooling'] = (-1) * (
        temp_set_c - outside_temp_c) * bus_window_area * window_ht_coeff_c / 1000
    q_heating.loc['q_wall', 'Heating'] = (-1) * (temp_set_h - outside_temp_h) * wall_area * wall_ht_coeff / 1000
    q_heating.loc['q_wall', 'Cooling'] = (-1) * (temp_set_c - outside_temp_c) * wall_area * wall_ht_coeff / 1000
    q_heating.loc['q_roof', 'Heating'] = (-1) * (temp_set_h - outside_temp_h) * roof_area * roof_ht_coeff / 1000
    q_heating.loc['q_roof', 'Cooling'] = (-1) * (temp_set_c - outside_temp_c) * roof_area * roof_ht_coeff / 1000
    q_heating.loc['q_floor', 'Heating'] = (-1) * (temp_set_h - outside_temp_h) * floor_area * floor_ht_coeff / 1000
    q_heating.loc['q_floor', 'Cooling'] = (-1) * (temp_set_c - outside_temp_c) * floor_area * floor_ht_coeff / 1000
    q_heating.loc['q_door', 'Heating'] = (-1) * heat_loss_door_open_h * stop_time * doors_per_bus / (
        time_between_stops + stop_time)
    q_heating.loc['q_door', 'Cooling'] = (-1) * heat_loss_door_open_c * stop_time * doors_per_bus / (
        time_between_stops + stop_time)

    q_pas_h, q_pas_c = q_heating.loc['q_pas', :'Cooling']
    q_sun_h, q_sun_c = q_heating.loc['q_sun', :'Cooling']
    q_ven_h, q_ven_c = q_heating.loc['q_ven', :'Cooling']
    q_window_h, q_window_c = q_heating.loc['q_window', :'Cooling']
    q_wall_h, q_wall_c = q_heating.loc['q_wall', :'Cooling']
    q_roof_h, q_roof_c = q_heating.loc['q_roof', :'Cooling']
    q_floor_h, q_floor_c = q_heating.loc['q_floor', :'Cooling']
    q_door_h, q_door_c = q_heating.loc['q_door', :'Cooling']

    if (BUS_TYPE == "be"):
        q_heating.loc['q_fc', 'Heating'] = 0
    elif (BUS_TYPE == "fc" or BUS_TYPE == "fc_h"):
        if ((-1) * (
            q_pas_h + q_sun_h + q_ven_h + q_window_h + q_wall_h + q_roof_h + q_floor_h + q_door_h) >= fuel_cell_heat_out_h):
            q_heating.loc['q_fc', 'Heating'] = fuel_cell_heat_out_h
        else:
            q_heating.loc['q_fc', 'Heating'] = (-1) * (
                q_pas_h + q_sun_h + q_ven_h + q_window_h + q_wall_h + q_roof_h + q_floor_h + q_door_h) * (
                                                   max_pcnt_total_heat_h / 100)

    q_fc_h = q_heating.loc['q_fc', 'Heating']

    if ((-1) * (
        q_pas_h + q_sun_h + q_ven_h + q_window_h + q_wall_h + q_roof_h + q_floor_h + q_door_h + q_fc_h) > min_hvac_load_h):
        q_heating.loc['q_hvac', 'Heating'] = (-1) * (
            q_pas_h + q_sun_h + q_ven_h + q_window_h + q_wall_h + q_roof_h + q_floor_h + q_door_h + q_fc_h)
    else:
        q_heating.loc['q_hvac', 'Heating'] = min_hvac_load_h

    if ((-1) * (
        q_pas_c + q_sun_c + q_ven_c + q_window_c + q_wall_c + q_roof_c + q_floor_c + q_door_c + 0) < min_hvac_load_c):
        q_heating.loc['q_hvac', 'Cooling'] = (-1) * (
            q_pas_c + q_sun_c + q_ven_c + q_window_c + q_wall_c + q_roof_c + q_floor_c + q_door_c + 0)
    else:
        q_heating.loc['q_hvac', 'Cooling'] = min_hvac_load_c

    setpoint = parameters.loc[('General Parameters', 'Temperature Setpoint'), 'Heating']

    q_heating.loc['hvac_req', 'Heating'] = q_heating.loc['q_hvac', 'Heating'] / 1
    q_heating.loc['hvac_req', 'Cooling'] = q_heating.loc['q_hvac', 'Cooling'] / 2.2

    hvac_total = 0

    if temperature >= setpoint:
        return round(abs(q_heating.loc['hvac_req', 'Cooling']), 2)
    else:
        return round(abs(q_heating.loc['hvac_req', 'Heating']), 2)
