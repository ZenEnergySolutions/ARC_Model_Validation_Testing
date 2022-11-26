import numpy as np

def ess_requirements_beb(dc, ess_pwr_max, ess_init_soc, ess_cap, fc_pwr_max, pwr_conv_eff, fc_time_full, fc_time_zero,
                         ess_round_trip_eff, ess_time_full, regen_eff_max, regen_a, regen_b, max_regen_pwr,
                         ess_soc_gain, ess_target_soc, ess_min_soc, ess_max_soc):
    from math import sqrt
    KPH_TO_MPH = 0.621371
    dc = dc.reset_index(drop=True)

    ess_pwr_dis_req = [0]
    ess_pwr_conv_dis_loss = [0]
    ess_max_dis_step = [0]
    ess_pwr_dis_ramp = [0]
    ess_pwr_dis_gross = [0]
    ess_pwr_dis_loss = [0]
    ess_pwr_dis_clip = [0]
    ess_soc_lim = [0]
    ess_pwr_dis_out = [ess_pwr_max]

    ess_pwr_chrg_req = [0]
    ess_pwr_conv_chrg_loss = [0]
    ess_pwr_chrg_net = [0]
    ess_pwr_regen_map = [0]
    ess_pwr_soc_lim = [0]
    ess_pwr_chrg_out = [0]
    ess_pwr_chrg_loss = [0]
    ess_pwr_chrg_clip = [0]
    mech_brk_pwr = [0]

    ess_soc = [ess_init_soc]
    ess_cur_cap = [ess_cap * ess_init_soc / 100]
    soc_error = [0]
    engine_adj_kw = [0]

    # dc["ess_cur_cap_dis"] =  0

    regen_pcnt = [0.005247]

    # from df
    ic_priority = np.array(dc.ic_priority)
    fc_priority = np.array(dc.fc_priority)
    ess_priority = np.array(dc.ess_priority)
    fc_pwr_deficit = np.array(dc.fc_pwr_deficit)
    pwr_trans_deccel = np.array(dc.pwr_trans_deccel)
    elec_pwr_out_chrg = np.array(dc.elec_pwr_out_chrg)
    ess_pwr_chrg_req = [0]
    elec_pwr_out_did = np.array(dc.elec_pwr_out_did)
    fc_pwr_out = np.array(dc.fc_pwr_out)
    ess_pwr_dis_req = [0]
    time_step = np.array(dc.time_step)
    speed = np.array(dc.speed)


    for i in range(dc.shape[0]):

        if i == 0:

            epcr_m1 = (ic_priority[i] == 1) & (ess_priority[i] > 0)
            epcr_m2 = ((fc_priority[i] == 1) & (ess_priority[i] > 0)) & (fc_pwr_deficit[i] <= 0)
            epcr_m3 = (ess_priority[i] == 1)
            ess_pwr_chrg_req_conditions = [epcr_m1, epcr_m2, epcr_m3]

            # ess_pwr_chrg_req values
            epcr_v1 = elec_pwr_out_chrg[i] / pwr_conv_eff
            epcr_v2 = fc_pwr_deficit[i] / pwr_conv_eff
            epcr_v3 = elec_pwr_out_chrg[i] / pwr_conv_eff
            ess_pwr_chrg_req_vals = [epcr_v1, epcr_v2, epcr_v3]

            ess_pwr_chrg_req = [np.select(ess_pwr_chrg_req_conditions, ess_pwr_chrg_req_vals, default=0)]
            ess_pwr_chrg_net = [ess_pwr_chrg_req[i] / (np.sqrt(ess_round_trip_eff))]
            ess_pwr_conv_chrg_loss = [ess_pwr_chrg_req[i] * (1 - pwr_conv_eff)]

            ess_m1 = (ic_priority[i] == 1) & (ess_priority[i] > 0)
            ess_m2 = ((fc_priority[i] == 1) & (ess_priority[i] > 0)) & (
                (elec_pwr_out_did[i] - fc_pwr_out[i]) <= 0)
            ess_m3 = ((fc_priority[i] == 1) & (ess_priority[i] > 0))
            ess_m4 = (ess_priority[i] == 1)
            ess_pwr_dis_req_conditions = [ess_m1, ess_m2, ess_m3, ess_m4]

            # ess_pwr_dis_req values
            ess_v1 = (elec_pwr_out_did[i] / pwr_conv_eff)
            ess_v2 = 0
            ess_v3 = ((elec_pwr_out_did[i] - fc_pwr_out[i]) / pwr_conv_eff)
            ess_v4 = (elec_pwr_out_did[i] / pwr_conv_eff)

            ess_pwr_dis_req_vals = [ess_v1, ess_v2, ess_v3, ess_v4]

            ess_pwr_dis_req = [np.select(ess_pwr_dis_req_conditions, ess_pwr_dis_req_vals, default=0)]
            ess_pwr_conv_dis_loss = [ess_pwr_dis_req[i] * (1 - pwr_conv_eff)]
            ess_max_dis_step = [time_step[i] * ess_pwr_max / ess_time_full]

            ess_pwr_dis_gross = [ess_pwr_dis_ramp[i] / np.sqrt(ess_round_trip_eff)]
            ess_pwr_dis_loss = [np.abs(dc.ess_pwr_dis_gross[i]) * (1 - np.sqrt(ess_round_trip_eff))]

            regen_pcnt = [regen_eff_max / (
                1 + regen_a * np.exp((-1 * regen_b) * (dc.speed[i] * KPH_TO_MPH + 1)))]

            ess_pwr_regen_map = [regen_pcnt[i] * ess_pwr_chrg_net[i]]

            ess_pwr_dis_out = [np.where(ess_pwr_max >= ess_soc_lim[i], ess_soc_lim[i], ess_pwr_max)]

            ess_pwr_soc_lim = [np.where(ess_pwr_regen_map[i] < 0, ess_pwr_regen_map[i], 0)]
            # dc["ess_pwr_soc_lim"] =  dc.ess_pwr_regen_map
            ess_pwr_chrg_out = [np.where(
                ((ess_priority[i] > 0) & (max_regen_pwr >= np.abs(ess_pwr_soc_lim[i]))), ess_pwr_soc_lim[i], 0)]

            ess_pwr_dis_clip = [ess_pwr_dis_gross[i] - ess_pwr_dis_out[i]]

            mech_brk_pwr = [pwr_trans_deccel[i] - ess_pwr_chrg_out[i]]
            ess_pwr_chrg_clip = [ess_pwr_chrg_net[i] - ess_pwr_chrg_out[i]]
            ess_pwr_chrg_loss = [abs(ess_pwr_chrg_out[i]) * (1 - sqrt(ess_round_trip_eff))]

            # dc.loc[ i, "ess_cur_cap_dis"] =  (dc.ess_pwr_chrg_out[i] + dc.ess_pwr_dis_out[i] ) * dc.time_step[i] / 3600

            ess_soc_lim = [np.where(ess_cur_cap[i] / ess_cap <= ess_min_soc, 0, ess_pwr_dis_gross[i])]
            ess_pwr_dis_out = [np.where(ess_pwr_max >= dc.ess_soc_lim[i], dc.ess_soc_lim[i], ess_pwr_max)]

            ess_soc = [ess_init_soc]

            ess_cur_cap = [ess_cap * ess_init_soc / 100]

            # Error in SOC vs Target
            soc_error = [ess_soc[i] - ess_target_soc]

            # engine_adj_kw conditions ~~~~in the model the SOC error is not percent, scale by 100 ~~~~~~~~~~~~~~~~~~~~VERIFY THAT THIS LOGIC IS CORRECT ~~~~~~~~~~~~~~~~~
            eak_m1 = soc_error[i] * ess_pwr_max * ess_soc_gain >= max_regen_pwr
            eak_m2 = soc_error[i] * max_regen_pwr * ess_soc_gain <= ((-1) * max_regen_pwr)
            engine_adj_kw_conditions = [eak_m1, eak_m2]

            # engine_adj_kw values
            eak_v1 = max_regen_pwr
            eak_v2 = (-1) * max_regen_pwr
            eak_default = soc_error[i] * max_regen_pwr * ess_soc_gain
            engine_adj_kw_vals = [eak_v1, eak_v2]

            # Engine Adjust Signal to Balance SOC
            # dc.loc[ i, "engine_adj_kw"] = np.select( engine_adj_kw_conditions, engine_adj_kw_vals, default = eak_default )
            if soc_error[i] * ess_pwr_max * ess_soc_gain >= max_regen_pwr:
                engine_adj = max_regen_pwr
            elif soc_error[i] * max_regen_pwr * ess_soc_gain <= ((-1) * max_regen_pwr):
                engine_adj = ((-1) * max_regen_pwr)
            else:
                engine_adj = soc_error[i] * max_regen_pwr * ess_soc_gain

            engine_adj_kw = [engine_adj]

        else:
            shift = i - 1

            # ESS DISCHARGE

            # ess_pwr_dis_req conditions
            ess_m1 = (ic_priority[i] == 1) & (ess_priority[i] > 0)
            ess_m2 = ((fc_priority[i] == 1) & (ess_priority[i] > 0)) & (
                (elec_pwr_out_did[i] - fc_pwr_out[i]) <= 0)
            ess_m3 = ((fc_priority[i] == 1) & (ess_priority[i] > 0))
            ess_m4 = (ess_priority[i] == 1)
            ess_pwr_dis_req_conditions = [ess_m1, ess_m2, ess_m3, ess_m4]

            # ess_pwr_dis_req values
            ess_v1 = (elec_pwr_out_did[i] / pwr_conv_eff)
            ess_v2 = 0
            ess_v3 = ((elec_pwr_out_did[i] - fc_pwr_out[i]) / pwr_conv_eff)
            ess_v4 = (elec_pwr_out_did[i] / pwr_conv_eff)

            ess_pwr_dis_req_vals = [ess_v1, ess_v2, ess_v3, ess_v4]

            ess_pwr_dis_req.append(np.select(ess_pwr_dis_req_conditions, ess_pwr_dis_req_vals, default=0))
            ess_pwr_conv_dis_loss.append(ess_pwr_dis_req[i] * (1 - pwr_conv_eff))
            ess_max_dis_step.append(time_step[i] * ess_pwr_max / ess_time_full)

            m1 = ((ess_pwr_dis_req[i] - ess_pwr_dis_out[shift]) >= 0) & (
                    (ess_pwr_dis_req[i] - ess_pwr_dis_out[shift]) > ess_max_dis_step[i])
            m2 = ((ess_pwr_dis_req[i] - ess_pwr_dis_out[shift]) >= 0)
            m3 = (ess_pwr_dis_req[i] >= ess_pwr_dis_out[shift])
            ess_pwr_dis_ramp_conds = [m1, m2, m3]

            v1 = ess_pwr_dis_out[shift] + ess_max_dis_step[i]
            v2 = ess_pwr_dis_out[shift] + (ess_pwr_dis_req[i] - ess_pwr_dis_out[shift])
            v3 = ess_pwr_dis_out[shift]
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

            ess_pwr_dis_ramp.append(np.select(ess_pwr_dis_ramp_conds, ess_pwr_dis_ramp_vals,
                                                      default=dc.ess_pwr_dis_out[shift]))

            ess_pwr_dis_gross.append(ess_pwr_dis_ramp[i] / np.sqrt(ess_round_trip_eff))
            ess_pwr_dis_loss.append(np.abs(ess_pwr_dis_gross[i]) * (1 - np.sqrt(ess_round_trip_eff)))
            ess_soc_lim.append(np.where(ess_soc[shift] <= ess_min_soc, 0, ess_pwr_dis_gross[i]))
            ess_pwr_dis_out.append(np.where(ess_pwr_max >= ess_soc_lim[i], ess_soc_lim[i], ess_pwr_max))
            ess_pwr_dis_clip.append(ess_pwr_dis_gross[i] - ess_pwr_dis_out[i])

            # if ((dc.ic_priority[i] == 1) & (dc.ess_priority[i] > 0)):
            #     ess_pwr_chrg_req = dc.elec_pwr_out_chrg[i] / pwr_conv_eff
            # elif (((dc.fc_priority[i] == 1) & (dc.ess_priority[i] > 0)) & (dc.fc_pwr_deficit[i] <= 0)):
            #     ess_pwr_chrg_req = dc.fc_pwr_deficit[i] / pwr_conv_eff
            # elif (dc.ess_priority[i] == 1):
            #     ess_pwr_chrg_req = dc.elec_pwr_out_chrg[i] / pwr_conv_eff
            # else:
            #     ess_pwr_chrg_req = 0
            m1 = ((ic_priority[i] == 1) & (ess_priority[i] > 0))
            m2 = (((fc_priority[i] == 1) & (ess_priority[i] > 0)) & (fc_pwr_deficit[i] <= 0))
            m3 = (ess_priority[i] == 1)
            v1 = elec_pwr_out_chrg[i] / pwr_conv_eff
            v2 = fc_pwr_deficit[i] / pwr_conv_eff
            v3 = elec_pwr_out_chrg[i] / pwr_conv_eff
            ess_pwr_chrg_req_conditions = [m1, m2, m3]
            ess_pwr_chrg_req_vals = [v1, v2, v3]

            ess_pwr_chrg_req.append(np.select(ess_pwr_chrg_req_conditions, ess_pwr_chrg_req_vals, default=0))
            ess_pwr_conv_chrg_loss.append(ess_pwr_chrg_req[i] * (1 - pwr_conv_eff))
            ess_pwr_chrg_net.append(ess_pwr_chrg_req[i] / (sqrt(ess_round_trip_eff)))

            ###Calculate the map values without needing the regen map
            regen_pcnt.append(regen_eff_max / (
                1 + regen_a * np.exp((-1 * regen_b) * (speed[i] * KPH_TO_MPH + 1))))
            ess_pwr_regen_map.append(regen_pcnt[i] * ess_pwr_chrg_net[i])

            if (ess_soc[shift] >= ess_max_soc):
                ess_pwr_soc_lim = 0
            else:
                ess_pwr_soc_lim = ess_pwr_regen_map[i]

            ess_pwr_soc_lim.append(ess_pwr_soc_lim)

            if (ess_priority[i] > 0 & (max_regen_pwr >= abs(ess_pwr_soc_lim[i]))):
                ess_pwr_chrg_out = ess_pwr_soc_lim[i]
            elif (ess_priority[i] > 0):
                ess_pwr_chrg_out = (-1) * max_regen_pwr
            else:
                ess_pwr_chrg_out = 0

            ess_pwr_chrg_out.append(ess_pwr_chrg_out)
            ess_pwr_chrg_loss.append(abs(ess_pwr_chrg_out[i]) * (1 - sqrt(ess_round_trip_eff)))
            ess_pwr_chrg_clip.append(ess_pwr_chrg_net[i] - ess_pwr_chrg_out[i])
            mech_brk_pwr.append(pwr_trans_deccel[i] - ess_pwr_chrg_out[i])

            ess_cur_cap.append(ess_cur_cap[shift] - (ess_pwr_dis_out[i] + ess_pwr_chrg_out[i]) * \
                                       time_step[i] / 3600)
            ess_soc.append(ess_cur_cap[i] / ess_cap * 100)
            soc_error.append(ess_soc[i] - ess_target_soc)  # * 100

            if ((soc_error[i] * ess_pwr_max * ess_soc_gain) >= max_regen_pwr):
                engine_adj = max_regen_pwr
            elif (soc_error[i] * max_regen_pwr * ess_soc_gain <= ((-1) * max_regen_pwr)):
                engine_adj = ((-1) * max_regen_pwr)
            else:
                engine_adj = soc_error[i] * max_regen_pwr * ess_soc_gain  # *100

            engine_adj_kw.append(engine_adj)  # np.select( engine_adj_kw_conditions, engine_adj_kw_vals, default = eak_default )

    return dc

