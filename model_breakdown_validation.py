"""
Zen Route Model

Function route_model takes a template object and extracts the constants. Then the fuction gets a duty in the form of a csv file, and a few static maps. The duty is transformed
to a pandas dataframe and functions are called to calculate force, power and energy values. The function calculates the each value at each time step. A results datafram is produced and a
sum summary of the data is produced to view the results of the duty cycle. 
"""

from model_calculations_validation import *  # create_csv, dc_forces, dc_power, aux_trans_power, engine_priorities,
# ice_power_limit, emotor_requirements, fuel_cell_init_beb, fuel_cell_requirements_beb,  ess_state_of_charge_beb,
# fuel_consumption, results_table, regen_map_calculations_beb , dashboard_graphs, sum_stats_and_format,
# fceb_fc_requirements, ess_requirements_beb
import cython
from test_helper import Timer
# from cython_build.ess_requirements import ess_req_beb_pyx


def route_model(model_input, duty_cycle, duty_cycle_name, temperature):
    timer = Timer()
    timer.start()
    dc = duty_cycle
    regen_map = pd.read_csv("./maps/regen_map.csv")
    emotor_eff_map = pd.read_csv("./maps/emotor_eff_map.csv")

    # USER INPUT & SELECTED CONSTANTS
    # ### Scientific Constants
    GRAVITY = 9.81  # gravitational constant
    RHO_AIR = 1.225

    # ### Force User Inputs and Constants
    gvw = model_input['gvw']  # 18327 #gross vehicle weight
    wheel_crr = model_input['wheel_crr']  # 0.007 #Wheel rolling resistance coefficient
    frontal_area = model_input['frontal_area']  # 8.783466304
    vehicle_cd = model_input['vehicle_cd']  # 0.8 #Drag coefficient

    # ### Auxilliary Power User Inputs and Constants
    pwr_aux_base = model_input['pwr_aux_base']  # 8
    pwr_aux_pcnt = model_input['pwr_aux_pcnt']  # 0
    pwr_conv_eff = model_input['pwr_conv_eff']  # 0.95

    # ### Transmission Power User Inputs and Constants
    trans_eff = model_input['trans_eff']  # 0.98

    # ### Drive Train Priorities and Engine Selection
    # backend reference values
    battery_engine = "be"
    ic_engine = ["ic", "ic_h"]
    fc_engine = ["fc", "fc_h"]
    hybrid = ["ic_h", "fc_h"]

    # Engine type for model
    engine_type = model_input['engine_type']
    ENGINE_SELECT = ""

    # ### IC Engine User Inputs and Constants
    ic_pwr_max = model_input['ic_pwr_max']  # 0
    ic_time_full = model_input['ic_time_full']  # 0
    ic_time_zero = model_input['ic_time_zero']  # 0

    # ### Electrical Motor User Inputs and Constants
    emotor_max_eff = model_input['emotor_max_eff']  # 0.95
    emotor_pwr_max = model_input['emotor_pwr_max']  # 380
    emotor_time_full = model_input['emotor_time_full']  # 4

    # ### Fuel Cell (FC) Engine User Inputs and Constants
    fc_pwr_max = model_input['fc_pwr_max']  # 0
    fc_time_full = model_input['fc_time_full']  # 0
    fc_time_zero = model_input['fc_time_zero']  # 0

    # ### Regen Map Calculation Constants and User Inputs
    regen_a = model_input['regen_a']  # 500
    regen_b = model_input['regen_b']  # 0.99
    regen_eff_max = model_input['regen_eff_max']  # 0.98 # Regen Breaking Efficiency
    max_regen_pwr = model_input['max_regen_pwr']  # 33

    # ### ESS User Inputs and Constants
    # ESS Power Limits
    pwr_conv_eff = model_input['pwr_conv_eff']  # 0.95 #( Power Converter Efficiency (%) )
    ess_pwr_max = model_input['ess_pwr_max']  # 380  #( Energy Storage Power (kW) )
    ess_round_trip_eff = model_input['ess_round_trip_eff']  # 0.97  #( Energy Storage Round Trip Efficiency (%) )
    ess_time_full = model_input['ess_time_full']  # 1 #( Energy Storage Time to Full Power (s) )
    ess_cap = model_input['ess_cap']  # 594 #( Energy Storage Capacity (kWh) )

    # ESS State of Charge (SOC)
    ess_init_soc = model_input['ess_init_soc']  # 0.95  #( ESS Initial State of charge (kWh) )
    ess_target_soc = model_input['ess_target_soc']  # 1 #Hybrid Battery SOC target
    ess_min_soc = model_input['ess_min_soc']  # 0.10 #( Battery min SOC (%) )
    ess_max_soc = model_input['ess_max_soc']  # 0.95 #( Battery max SOC (%) )
    ess_soc_gain = model_input['ess_soc_gain']  # 0.3 #SOC control proportional gain

    # ### Fuel Type User Inputs and Constants
    gas_lhv = 12.22  # Gasoline Heating Value (LHV)
    diesel_lhv = 11.96  # Diesel Heating Value (LHV)
    ng_lhv = 13.03  # Natural Gas Heating Value (LHV)
    h2_lhv = 33.32  # Hydrogen Heating Value (LHV)
    gas_density = 0.737  # Gasoline Density
    diesel_density = 0.832  # Diesel Density
    ng_density = 0.000733  # Natural Gas Density (STP)
    h2_density = 0.000899  # Hydrogen Density
    diesel_gge = 1.155  # GGE Per Gal Diesel
    cng_gge = 0.00108  # GGE per Gal CNG @STP
    h2_gge = 1.019  # Hydrogen kg per gallon of gasoline
    elec_gge = 0.031  # GGE per kWh of electricity
    FUEL_TANK_SIZE = 38  # kg
    timer.stop()
    print('Time for variable assignment is {:.2f}s'.format(timer.elapsed_time))
    timer.start()
    dc['cycle'] = duty_cycle_name

    temperature = float(temperature)

    hvac_load = hvac_heating(engine_type, temperature)

    timer.stop()
    print('Time for hvac calcs is {:.2f}s'.format(timer.elapsed_time))
    timer.start()

    dc = dc_forces(dc, gvw, GRAVITY, RHO_AIR, frontal_area, vehicle_cd, wheel_crr)

    timer.stop()
    print('Time for dc_forces calcs is {:.2f}s'.format(timer.elapsed_time))
    timer.start()

    dc = dc_power(dc)

    timer.stop()
    print('Time for dc_power calcs is {:.2f}s'.format(timer.elapsed_time))
    timer.start()

    dc = aux_trans_power(dc, pwr_aux_base, pwr_aux_pcnt, pwr_conv_eff, trans_eff, hvac_load)

    timer.stop()
    print('Time for aux_trans_power calcs is {:.2f}s'.format(timer.elapsed_time))
    timer.start()

    dc = engine_priorities(dc, engine_type, ic_engine, fc_engine, hybrid)

    timer.stop()
    print('Time for engine_priorities calcs is {:.2f}s'.format(timer.elapsed_time))
    timer.start()

    dc = ice_power_limit(dc, ic_pwr_max, ic_time_full, ic_time_zero)

    timer.stop()
    print('Time for ice_power_limit calcs is {:.2f}s'.format(timer.elapsed_time))
    timer.start()

    dc = emotor_requirements(dc, emotor_eff_map, emotor_pwr_max, emotor_max_eff, emotor_time_full)

    timer.stop()
    print('Time for emotor_requirements calcs is {:.2f}s'.format(timer.elapsed_time))
    timer.start()

    if engine_type == 'be':
        print('BE Selected')
        dc = fuel_cell_init_beb(dc, pwr_conv_eff, fc_pwr_max, fc_time_full, fc_time_zero)

        timer.stop()
        dc.to_csv('ess_req_test.csv', index=False)
        print('Time for fuel_cell_init_beb calcs is {:.2f}s'.format(timer.elapsed_time))
        timer.start()

        dc = ess_requirements_beb(dc, ess_pwr_max, ess_init_soc, ess_cap,
                                  fc_pwr_max, pwr_conv_eff, fc_time_full, fc_time_zero,
                                  ess_round_trip_eff, ess_time_full, regen_eff_max, regen_a,
                                  regen_b, max_regen_pwr, ess_soc_gain, ess_target_soc, ess_min_soc, ess_max_soc)

        # dc = ess_req_beb_pyx(dc, ess_pwr_max, ess_init_soc, ess_cap,
        #                           fc_pwr_max, pwr_conv_eff, fc_time_full, fc_time_zero,
        #                           ess_round_trip_eff, ess_time_full, regen_eff_max, regen_a,
        #                           regen_b, max_regen_pwr, ess_soc_gain, ess_target_soc, ess_min_soc, ess_max_soc)

        timer.stop()
        print('Time for ess_requirements_beb calcs is {:.2f}s'.format(timer.elapsed_time))


    else:
        print('FCEB FC Requirements Selected')
        dc = fceb_fc_requirements(dc, ess_pwr_max, ess_init_soc, ess_cap, fc_pwr_max, pwr_conv_eff,
                                  fc_time_full, fc_time_zero, ess_round_trip_eff, ess_time_full, regen_eff_max,
                                  regen_a, regen_b, max_regen_pwr, ess_soc_gain, ess_target_soc, ess_min_soc,
                                  ess_max_soc)
        timer.stop()
        print('Time for fceb_requirements calcs is {:.2f}s'.format(timer.elapsed_time))

    timer.start()

    dc = fuel_consumption(dc, ic_pwr_max, fc_pwr_max, h2_lhv)

    timer.stop()
    print('Time for fuel consumption calcs is {:.2f}s'.format(timer.elapsed_time))
    timer.start()

    results = results_table(dc, ess_round_trip_eff, pwr_conv_eff, trans_eff)

    dc_script, dc_div, ess_script, ess_div = dashboard_graphs(dc, ess_min_soc, engine_type, FUEL_TANK_SIZE)

    dc, results_stats, sum_table = sum_stats_and_format(dc, results)

    # #tranpose summary table
    # sum_table = sum_table.T

    ### Print out csv's
    # create_csv(dc, results)

    # Metrics
    distance = round(dc['distance'].max(), 2)
    duration = round(dc['time'].max() / 3600, 0)
    mean_grade = round(dc['grade'].mean(), 2)
    max_grade = round(dc['grade'].max(), 2)
    ave_speed = round(dc['speed'].mean(), 2)

    ### Results

    if engine_type == 'be':
        if dc[dc['ess_soc'] <= ess_min_soc].empty:
            complete = 100
            distance_complete = distance
            try:
                fuel_consumed = round((ess_init_soc - dc['ess_soc'].min()) * ess_cap / 100, 2)
                fuel_economy = round(fuel_consumed / distance_complete, 2)
            except ZeroDivisionError:
                fuel_consumed = 0
                fuel_economy = 0
        else:
            complete = round((dc[dc['ess_soc'] <= ess_min_soc]['time'].min() / 3600) / duration * 100)
            distance_complete = round(complete * distance / 100, 2)
            try:
                fuel_consumed = round((ess_init_soc - ess_min_soc) * ess_cap / 100, 2)
                fuel_economy = round(fuel_consumed / distance_complete, 2)
            except ZeroDivisionError:
                fuel_consumed = 0
                fuel_economy = 0

    else:
        if dc['h2_consumption'].max() <= FUEL_TANK_SIZE:
            complete = 100
            distance_complete = distance
        else:
            complete = round(dc[dc['h2_consumption'] >= FUEL_TANK_SIZE]['time'].min() / 3600)
            distance_complete = round(complete * distance / 100, 2)
        try:
            fuel_consumed = round(dc['h2_consumption'].max(), 2)
            fuel_economy = round(fuel_consumed / distance_complete, 2)
        except ZeroDivisionError:
            fuel_consumed = 0
            fuel_economy = 0

    results = results.set_index(dc['time'])
    results.index.name = 'Time (s)'

    results_dt = results.sum()
    results_dt = results_dt.round(2).to_dict()

    timer.stop()
    print('Time for results calcs is {:.2f}'.format(timer.elapsed_time))

    return dc, results, results_dt, sum_table, distance, duration, mean_grade, max_grade, ave_speed, complete, \
           distance_complete, fuel_consumed, fuel_economy, engine_type
